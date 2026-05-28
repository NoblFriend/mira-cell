from __future__ import annotations

from pathlib import Path

import fire
import numpy as np
import torch
from omegaconf import OmegaConf

from mira_cell.constants import NUM_CLASSES
from mira_cell.models.classifier import LetterClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_CONFIG = REPO_ROOT / "configs" / "model" / "resnet18_hint.yaml"
_DUMMY_OPTIM = {"lr": 0.0, "weight_decay": 0.0, "scheduler_t_max": 1, "scheduler_eta_min": 0.0}


def _load_model(checkpoint: str, model_config: str) -> LetterClassifier:
    cfg = OmegaConf.load(model_config)
    cfg = OmegaConf.merge(cfg, OmegaConf.create(_DUMMY_OPTIM))
    model = LetterClassifier.load_from_checkpoint(checkpoint, cfg=cfg, map_location="cpu")
    model.eval()
    return model


def _onnx_session(onnx_path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def _verify(model: LetterClassifier, onnx_path: Path, num_channels: int, img_size: int, atol: float) -> float:
    image = torch.randn(4, num_channels, img_size, img_size)
    hint_mask = torch.ones(4, NUM_CLASSES)
    with torch.inference_mode():
        torch_logits = model(image, hint_mask).numpy()
    session = _onnx_session(onnx_path)
    onnx_logits = session.run(
        ["logits"],
        {"image": image.numpy(), "hint_mask": hint_mask.numpy()},
    )[0]
    max_diff = float(np.abs(torch_logits - onnx_logits).max())
    if max_diff > atol:
        raise AssertionError(f"ONNX/torch mismatch: max abs diff {max_diff:.3e} > atol {atol:.3e}")
    return max_diff


def export_onnx(
    checkpoint: str,
    output: str = "models/model.onnx",
    model_config: str = str(DEFAULT_MODEL_CONFIG),
    img_size: int = 128,
    num_channels: int = 3,
    opset: int = 17,
    atol: float = 1e-4,
) -> str:
    import onnx

    model = _load_model(checkpoint, model_config)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_image = torch.randn(1, num_channels, img_size, img_size)
    dummy_hint = torch.ones(1, NUM_CLASSES)

    torch.onnx.export(
        model,
        (dummy_image, dummy_hint),
        str(output_path),
        input_names=["image", "hint_mask"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch"},
            "hint_mask": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=opset,
    )

    onnx.checker.check_model(onnx.load(str(output_path)))
    max_diff = _verify(model, output_path, num_channels, img_size, atol)
    print(f"[export] wrote {output_path} (opset {opset})")
    print(f"[export] sanity check passed: max |torch - onnx| = {max_diff:.3e}")
    return str(output_path)


def to_mlflow(
    checkpoint: str,
    output: str = "models/model.onnx",
    model_config: str = str(DEFAULT_MODEL_CONFIG),
    tracking_uri: str = "http://127.0.0.1:8080",
    experiment_name: str = "mira-cell-serving",
    artifact_path: str = "model",
    img_size: int = 128,
    num_channels: int = 3,
    opset: int = 17,
    atol: float = 1e-4,
) -> str:
    import mlflow
    import onnx
    from mlflow.models import infer_signature

    onnx_path = export_onnx(
        checkpoint=checkpoint,
        output=output,
        model_config=model_config,
        img_size=img_size,
        num_channels=num_channels,
        opset=opset,
        atol=atol,
    )

    sample_input = {
        "image": np.random.randn(1, num_channels, img_size, img_size).astype(np.float32),
        "hint_mask": np.ones((1, NUM_CLASSES), dtype=np.float32),
    }
    sample_output = _onnx_session(Path(onnx_path)).run(["logits"], sample_input)[0]
    signature = infer_signature(sample_input, sample_output)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="onnx-export") as run:
        mlflow.onnx.log_model(
            onnx_model=onnx.load(onnx_path),
            artifact_path=artifact_path,
            signature=signature,
        )
        model_uri = f"runs:/{run.info.run_id}/{artifact_path}"

    print(f"[mlflow] logged ONNX model: {model_uri}")
    print(f"[mlflow] serve with: mlflow models serve -m {model_uri} -p 5001 --env-manager local")
    return model_uri


def cli() -> None:
    fire.Fire({"onnx": export_onnx, "mlflow": to_mlflow})


if __name__ == "__main__":
    cli()
