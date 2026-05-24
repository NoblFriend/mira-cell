from __future__ import annotations

from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from form_reader_and_evaluator.data.datamodule import NISTDataModule
from form_reader_and_evaluator.models.classifier import LetterClassifier
from form_reader_and_evaluator.utils.download import download_data
from form_reader_and_evaluator.utils.git_meta import current_commit_id

REPO_ROOT = Path(__file__).resolve().parents[1]


def _flat(cfg: DictConfig) -> dict:
    container = OmegaConf.to_container(cfg, resolve=True)

    def walk(prefix: str, node) -> dict:
        out: dict = {}
        if isinstance(node, dict):
            for k, v in node.items():
                key = f"{prefix}.{k}" if prefix else k
                out.update(walk(key, v))
        else:
            out[prefix] = node
        return out

    return walk("", container)


def _make_logger(cfg: DictConfig) -> MLFlowLogger:
    return MLFlowLogger(
        experiment_name=cfg.logger.experiment_name,
        tracking_uri=cfg.logger.tracking_uri,
        run_name=cfg.run_name,
        log_model=cfg.logger.log_model,
    )


def _make_model(cfg: DictConfig) -> LetterClassifier:
    merged = OmegaConf.merge(cfg.model, cfg.optimizer)
    return LetterClassifier(merged)


def _make_trainer(
    cfg: DictConfig,
    logger: MLFlowLogger,
    checkpoints_dir: Path,
) -> L.Trainer:
    monitor = "val/acc_hint_AH"
    ckpt = ModelCheckpoint(
        dirpath=str(checkpoints_dir),
        monitor=monitor,
        mode="max",
        save_top_k=2,
        save_last=True,
        auto_insert_metric_name=False,
        filename="epoch={epoch:02d}-AH={" + monitor + ":.4f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    return L.Trainer(
        default_root_dir=str(REPO_ROOT / "lightning_logs"),
        accelerator=cfg.trainer.accelerator,
        precision=cfg.trainer.precision,
        max_epochs=cfg.trainer.max_epochs,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        limit_train_batches=cfg.trainer.limit_train_batches or 1.0,
        limit_val_batches=cfg.trainer.limit_val_batches or 1.0,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        deterministic=cfg.trainer.deterministic,
        callbacks=[ckpt, lr_monitor],
        logger=logger,
    )


def train(cfg: DictConfig) -> None:
    L.seed_everything(cfg.seed, workers=True)

    download_data(target=cfg.paths.data_dvc_target, repo_root=str(REPO_ROOT))

    checkpoints_dir = REPO_ROOT / cfg.paths.checkpoints_dir
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    logger = _make_logger(cfg)
    logger.log_hyperparams(_flat(cfg))
    logger.experiment.log_param(logger.run_id, "git_commit", current_commit_id(REPO_ROOT))

    datamodule = NISTDataModule(cfg.data)
    model = _make_model(cfg)
    trainer = _make_trainer(cfg, logger=logger, checkpoints_dir=checkpoints_dir)

    trainer.fit(model, datamodule=datamodule)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    train(cfg)


def cli() -> None:
    main()


if __name__ == "__main__":
    main()
