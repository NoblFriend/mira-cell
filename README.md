# Form reader and evaluator

Распознавание рукописной буквы в одной ячейке бланка — модуль для автоматизации
проверки **блиц-туров Всероссийской олимпиады по астрономии**. Модель работает
как часть конвейера `restore → recognize → grade` и предназначена для того,
чтобы минимизировать число ошибок распознавания, а сомнительные случаи
эффективно отправлять на ручную пост-проверку — и тем самым сократить
количество апелляций.

Ключевая особенность — **распознавание с подсказкой о допустимых буквах**:
вместе с изображением модели передаётся бинарная маска по 28 классам, что
сужает пространство возможных ответов и заметно повышает надёжность результата
в реальном сценарии проведения тура.

---

## Постановка задачи

Классификация изображения одной ячейки бланка на **28 классов**:

- `A`..`Z` — 26 заглавных латинских букв
- `<empty>` — ячейка пустая
- `<junk>` — нечитаемая / зачёркнутая запись

### Формат входа

- Изображение ячейки, RGB, **128×128**, PNG (после восстановления геометрии
  сканера в модуле `restore-coded`).
- Опциональная подсказка — список допустимых букв (например, `["A","B","C","D","E","F"]`),
  передаётся внутрь модели в виде бинарной маски размерности 28.

### Формат выхода

- Предсказанный класс (`A`..`Z`, `<empty>`, `<junk>`)
- `confidence` — вероятность top-1 после softmax
- `margin` — разница между top-1 и top-2 (используется как сигнал к ручной
  пере-проверке)

### Метрики

| Метрика                  | Бейзлайн (CNN без подсказки) | Целевой ориентир | Достигнуто на real-scan (3705 ячеек) |
|--------------------------|------------------------------|------------------|--------------------------------------|
| `val/acc_no_hint`        | ≥ 0.97                       | ≥ 0.99           | **0.9943**                           |
| `val/acc_hint_AH`        | ≥ 0.97                       | ≥ 0.995          | **0.9989**                           |
| Доля примеров на ручную проверку при нулевых ошибках | — | минимально возможная | < 1%                                 |

Сценарий применения: даже единичные ошибки на 12 000 ячеек тура приводят к
росту апелляций, поэтому цель — после ручной пост-проверки спорных кейсов
получать **ноль ошибок** распознавания.

---

## Данные

| Источник                                                                                          | Тип                    | Размер             | Дата выпуска |
| ------------------------------------------------------------------------------------------------- | ---------------------- | ------------------ | ------------ |
| [NIST Special Database 19](https://www.nist.gov/srd/nist-special-database-19) (by_class, A–Z)     | Рукописные глифы букв  | ~300 МБ, ~300k img | 1995, ред. 2016 |
| Синтетика для `<empty>` / `<junk>`                                                                | Сгенерировано локально | ~10 МБ             | 2026         |
| Исторические сканы тренировочных туров (вне публичного датасета)                                  | Доменные real-scan     | ~50 МБ             | 2024–2026    |

Структура `data/`:

```
data/
├── train/<class>/*.png   # ImageFolder, 28 классов
├── val/<class>/*.png
└── cell_variants/*.png   # 10 заготовок рамки клетки 94×94, накладываются на лету
```

**Особенности данных и риски**

- сильный дисбаланс классов (буквы A–H встречаются чаще)
- ошибки разметки в NIST → планируется чистка по предсказаниям бейзлайна
- доменный сдвиг между NIST-глифами и реальными сканами тура (закрывается
  стадией finetune на размеченных real-scan ячейках)
- сейчас train использует только синтетические композиции «NIST-глиф + рамка»

Данные хранятся в **DVC** (см. раздел *Setup*); пример входной ячейки лежит
рядом с весами в DVC-хранилище под `data/examples/cell.png`.

---

## Моделирование

### Архитектура

- **Backbone**: `torchvision.models.resnet18` с весами `ResNet18_Weights.DEFAULT`,
  выход 512-мерных фичей.
- **Hint encoder**: MLP `28 → 128 → 128` поверх бинарной маски подсказки.
- **Fusion**: проекции image и hint в общее пространство `128`, перемножение
  поэлементно (`p_img ⊙ p_hint`), затем конкатенация
  `[f_img (512), f_hint (128), interaction (128)]` → MLP-голова
  `768 → 512 → 256 → 28`.
- **Hint residual**: лёгкая ветка `28 → 64 → 28` с обучаемым `alpha`
  (init `0.05`), добавляется к логитам — даёт модели прямой шорткат «знаешь
  букву — учитывай её».

### Бейзлайн

CNN-классификатор без подсказки (тот же ResNet18 без hint-веток), обучение на
тех же синтетических композициях NIST + рамка. Используется как нижняя
граница и как инструмент для чистки разметки.

### Стек

- **PyTorch 2.x** + **torchvision** (модели, transforms, ImageFolder)
- **PyTorch Lightning** (`L.Trainer`, `L.LightningModule`, `L.LightningDataModule`,
  колбэки `ModelCheckpoint`, `LearningRateMonitor`)
- **Hydra** для конфигов и точки входа
- **MLflow** для логирования метрик и гиперпараметров
- **DVC** (gdrive remote) для данных и весов
- **ruff** + **pre-commit** для качества кода

### Формат модели

- Промежуточные снимки: Lightning `.ckpt` (хранится в DVC, не в git).
- Прод-формат: **ONNX** (см. Task 3), опционально TensorRT.

### Ресурсы и латентность

- Обучение: 1 GPU класса T4 / Apple M-series MPS; полный прогон 120 эпох ~2–3 ч.
- Инференс батчами: на CPU ~50 ячеек/с в ONNX runtime; этого хватает, потому
  что один тур — это ~12 000 ячеек, целевая длительность инференса < 5 минут.

---

## Setup

Используется **poetry** (зависимости в `pyproject.toml`, lock-файл `poetry.lock`).

```bash
git clone <repo-url>
cd form-reader-and-evaluator

# 1. зависимости
poetry install
poetry shell

# 2. pre-commit
pre-commit install
pre-commit run -a

# 3. данные (DVC remote — gdrive)
form-download-data --target=data
# или вручную:  dvc pull data
```

> Альтернатива poetry — `uv sync` (с тем же `pyproject.toml`).
> В `.dvc/config` прописаны два remote: `data` (по умолчанию) и `models` —
> для весов.

---

## Train

Единая точка входа через Hydra:

```bash
# полный прогон
form-train

# смоук-тест за минуту (1 эпоха, 5 батчей)
form-train trainer=smoke

# переопределение отдельных параметров
form-train data.batch_size=128 optimizer.lr=1e-4 trainer.max_epochs=30
```

Что делает `train`:

1. Дергает `download_data()` → `dvc pull` нужных таргетов.
2. Поднимает `MLFlowLogger` (адрес — в `configs/logger/mlflow.yaml`,
   по умолчанию `http://127.0.0.1:8080`).
3. Логирует все hydra-параметры одной плоской пачкой + git commit id
   (`git rev-parse HEAD`).
4. Запускает `L.Trainer.fit(model, datamodule)`.
5. Чекпойнты сохраняются в `checkpoints/`; лучшие отбираются по
   `val/acc_hint_AH`.

Графики обучения предыдущего рана (для предзагрузки в `plots/`) — лосс,
accuracy и `lr / alpha` — лежат в [plots/](plots/):

- [plots/loss.png](plots/loss.png)
- [plots/accuracy.png](plots/accuracy.png)
- [plots/lr_alpha.png](plots/lr_alpha.png)

---

## Infer

Точка входа — [form_reader_and_evaluator/infer.py](form_reader_and_evaluator/infer.py)
(минимум зависимостей, без тренировочного стека):

```bash
form-infer \
  --image_path=data/examples/cell.png \
  --checkpoint=checkpoints/best.ckpt \
  --allowed=ABCDEF
```

Полная подготовка модели к проду (ONNX-экспорт, Triton-сервер) — задача Task 3.

---

## Overall

```
form-reader-and-evaluator/
├── form_reader_and_evaluator/      # python-пакет (snake_case)
│   ├── __init__.py
│   ├── constants.py                # CLASS_NAMES, LETTER_TO_IDX, …
│   ├── train.py                    # hydra entry point
│   ├── infer.py                    # lean inference CLI
│   ├── data/
│   │   ├── datamodule.py           # NISTDataModule (LightningDataModule)
│   │   ├── hint.py                 # sample_hint_mask, allowed_letters_to_mask
│   │   └── transforms.py           # AddFrameTensor + build_transforms
│   ├── models/
│   │   └── classifier.py           # LetterClassifier (LightningModule)
│   └── utils/
│       ├── download.py             # dvc pull wrapper
│       └── git_meta.py             # current_commit_id()
├── configs/                        # hydra, hierarchical
│   ├── config.yaml                 # defaults, единственная точка входа
│   ├── data/nist.yaml
│   ├── model/resnet18_hint.yaml
│   ├── optimizer/adamw_cosine.yaml
│   ├── trainer/{default,smoke}.yaml
│   └── logger/mlflow.yaml
├── plots/                          # графики обучения (PNG)
├── scripts/                        # вспомогательные скрипты
├── .dvc/config                     # два remote: data, models
├── .pre-commit-config.yaml         # pre-commit-hooks + ruff + prettier
├── pyproject.toml                  # poetry + ruff settings
├── poetry.lock
└── README.md
```

### Пайплайн и внедрение

```
PDF скан тура ──► restore ──► ячейки 128×128 ──► recognize (этот репо) ──► grade
                                                       │
                                                       ├─ label, confidence, margin
                                                       └─ low-confidence → ручная проверка
```

Модуль вызывается из общего пайплайна `form-reader-pipeline`
(см. родительский репозиторий конвейера), внутри тура работает в пакетном
режиме на одном GPU/CPU; модель сериализуется в ONNX для продакшена.

---

## Changes since Task 1 proposal

По итогам ревью первого этапа доработано:

- название проекта закреплено как `form-reader-and-evaluator`
- технологический стек выписан явно (PyTorch / Lightning / Hydra / MLflow / DVC)
- зафиксирован формат входных данных: PNG, 128×128, 3 канала, RGB
- метрики приведены с числовыми целями и реально достигнутыми значениями на
  3705 размеченных вручную ячейках real-scan
- описаны датасеты: размеры, даты, способ сбора синтетики
- разобран бейзлайн и основная архитектура с конкретными слоями и числами
- дописан формат модели для прод-внедрения (ONNX, далее TensorRT) и оценка
  ресурсов и латентности
