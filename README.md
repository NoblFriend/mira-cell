# Mira Cell

**Mira Cell** — распознавание рукописной буквы в одной ячейке бланка для
автоматизации проверки **блиц-туров Всероссийской олимпиады по астрономии**.
Модель работает
как часть конвейера `restore → recognize → grade` и предназначена для того,
чтобы минимизировать число ошибок распознавания, а сомнительные случаи
эффективно отправлять на ручную пост-проверку — и тем самым сократить
количество апелляций.

Ключевая особенность — **распознавание с подсказкой о допустимых буквах**:
вместе с изображением модели передаётся бинарная маска по 28 классам, что
сужает пространство возможных ответов и заметно повышает надёжность результата
в реальном сценарии проведения тура.

> Модель уже применялась в этом году на финале ВсОШ по астрономии. Впервые
> за всю историю олимпиады по итогам блиц-туров не поступило ни одной
> апелляции, а полная проверка и пере-проверка ответов заняла не более двух
> часов — раньше на тот же объём уходили часы работы нескольких человек.

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

| Метрика                                              | Бейзлайн (CNN без подсказки) | Целевой ориентир     | Достигнуто на размеченных вручную real-scan ячейках |
| ---------------------------------------------------- | ---------------------------- | -------------------- | --------------------------------------------------- |
| `val/acc_no_hint`                                    | ≥ 0.97                       | ≥ 0.99               | **0.9943**                                          |
| `val/acc_hint_AH`                                    | ≥ 0.97                       | ≥ 0.995              | **0.9989**                                          |
| Доля примеров на ручную проверку при нулевых ошибках | —                            | минимально возможная | < 1%                                                |

Сценарий применения: даже единичные ошибки на 12 000 ячеек тура приводят к
росту апелляций, поэтому цель — после ручной пост-проверки спорных кейсов
получать **ноль ошибок** распознавания.

---

## Данные

| Источник                                                                                      | Тип                    | Размер             | Дата выпуска    |
| --------------------------------------------------------------------------------------------- | ---------------------- | ------------------ | --------------- |
| [NIST Special Database 19](https://www.nist.gov/srd/nist-special-database-19) (by_class, A–Z) | Рукописные глифы букв  | ~300 МБ, ~300k img | 1995, ред. 2016 |
| Синтетика для `<empty>` / `<junk>`                                                            | Сгенерировано локально | ~10 МБ             | 2026            |

Исторические сканы реальных бланков олимпиады в обучении не используются —
прав на их публичное распространение нет, в открытый DVC-remote они не
выкладываются. Хранятся локально и применяются только для финальной валидации
качества на доменных данных.

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
- доменный сдвиг между NIST-глифами и реальными сканами тура снимается на
  препроцессинге: тоновая кривая `[35, 210] → [0, 255]` приводит фон к
  одинаково белому, а штрих — к насыщенно чёрному, что выравнивает
  распределение интенсивностей
- train использует только синтетические композиции «NIST-глиф + рамка
  ячейки» — реальные сканы остаются исключительно для офлайн-проверки

Данные хранятся в **DVC** (см. раздел _Setup_) и доступны проверяющему
через публичный GitHub Release этого репозитория.

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
- **Hint residual**: лёгкая ветка `28 → 64 → 28`, выход которой
  буквально прибавляется к финальным логитам и обучаемо сдвигает их в
  пользу разрешённых классов. Перед сложением умножается на отдельный
  обучаемый скаляр `alpha` (init `0.05`): одно дело — _как именно_ ветка
  сдвигает логиты, другое — _какой общий вес_ ей дать в сумме с остальной
  моделью. `alpha` вынесен в отдельный параметр и логируется как
  `train/alpha`, чтобы можно было следить за этим балансом.

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
- **DVC** (локальный remote + публичный GitHub Release для дистрибуции)
- **ruff** + **pre-commit** для качества кода

### Формат модели

- Промежуточные снимки: Lightning `.ckpt` (хранится в DVC, не в git).
- Прод-формат: **ONNX**.

### Ресурсы и латентность

- Обучение: фактически велось на **Apple MacBook Pro M1 Max** (MPS), один полный
  прогон 120 эпох ~2–3 часа; запуск без изменений переносится на одиночный
  GPU класса T4.
- Инференс батчами: на CPU ~50 ячеек/с в ONNX runtime; на один тур (~12 000
  ячеек) — единицы минут.

---

## Setup

Используется [**uv**](https://github.com/astral-sh/uv) — менеджер окружения
и зависимостей. Зависимости в `pyproject.toml`, lock-файл `uv.lock`.

```bash
git clone <repo-url>
cd mira-cell

# 1. зависимости + venv одной командой
uv sync

# 2. pre-commit
uv run pre-commit install
uv run pre-commit run -a

# 3. данные и веса — публичный GitHub Release, никаких creds не нужно
uv run mira-download-data --target=data    # data/{train,val,cell_variants}
uv run mira-download-data --target=models  # models/best.ckpt
```

В `.dvc/config` прописаны два remote: `data` и `models` — оба локальные
(`../mira-cell-dvc-storage/{data,models}`), нужны только для проверки
структуры. Реальная дистрибуция данных и весов идёт через
`mira-download-data`, которая стрим-качает `tar.gz` с релиза этого репо и
распаковывает в `data/` / `models/`. Поэтому проверяющий запускает
`mira-train` без каких-либо предварительных авторизаций.

---

## Train

Единая точка входа через Hydra:

```bash
# полный прогон
uv run mira-train

# смоук-тест за минуту (1 эпоха, 5 батчей)
uv run mira-train trainer=smoke

# переопределение отдельных параметров
uv run mira-train data.batch_size=128 optimizer.lr=1e-4 trainer.max_epochs=30
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

- [plots/loss.png](plots/loss.png) — train/val loss
- [plots/accuracy.png](plots/accuracy.png) — train/val accuracy, no-hint и hint-AH
- [plots/learning_rate.png](plots/learning_rate.png) — расписание lr (warmup → plateau)
- [plots/hint_alpha.png](plots/hint_alpha.png) — обучаемый коэффициент при hint-residual

---

## Infer

Точка входа — [mira_cell/infer.py](mira_cell/infer.py)
(минимум зависимостей, без тренировочного стека):

```bash
uv run mira-infer \
  --image_path=data/examples/cell.png \
  --checkpoint=checkpoints/best.ckpt \
  --allowed=ABCDEF
```

---

## Production preparation

### Экспорт в ONNX

Прод-формат модели — **ONNX**. Экспорт берёт Lightning `.ckpt` и пишет
граф с двумя входами (`image` `[B,3,128,128]`, `hint_mask` `[B,28]`) и
выходом `logits` `[B,28]`; batch — динамическая ось. После записи граф
проверяется `onnx.checker` и сверяется с PyTorch (max abs diff логитов):

```bash
uv run mira-export onnx \
  --checkpoint=models/best.ckpt \
  --output=models/model.onnx
```

Препроцессинг (resize/normalize) и постпроцессинг (softmax → top-1,
`confidence`, `margin`) остаются вне графа — на стороне клиента (см.
`scripts/query_server.py`), чтобы граф был переносимым и лёгким.

## Infer через сервер (MLflow Serving)

Регистрация ONNX-модели в MLflow (тот же tracking-сервер
`http://127.0.0.1:8080`) с подписью входов/выходов:

```bash
uv run mira-export mlflow --checkpoint=models/best.ckpt
# -> печатает model_uri вида runs:/<run_id>/model
```

Поднять REST-сервер инференса и постучаться тест-клиентом:

```bash
mlflow models serve -m runs:/<run_id>/model -p 5001 --env-manager local

uv run python scripts/query_server.py \
  --image_path=data/val/A/<example>.png \
  --allowed=ABCDEF
```

Клиент препроцессит PNG, шлёт `POST /invocations` и печатает
`{label, confidence, margin}`.

---

## Overall

```
mira-cell/
├── mira_cell/                      # python-пакет (snake_case)
│   ├── __init__.py
│   ├── constants.py                # CLASS_NAMES, LETTER_TO_IDX, …
│   ├── train.py                    # hydra entry point
│   ├── infer.py                    # lean inference CLI
│   ├── export.py                   # ONNX export + MLflow model registration
│   ├── data/
│   │   ├── datamodule.py           # NISTDataModule (LightningDataModule)
│   │   ├── hint.py                 # sample_hint_mask, allowed_letters_to_mask
│   │   └── transforms.py           # AddFrameTensor + build_transforms
│   ├── models/
│   │   └── classifier.py           # LetterClassifier (LightningModule)
│   └── utils/
│       ├── download.py             # stream-download tar.gz из GitHub Release
│       └── git_meta.py             # current_commit_id()
├── configs/                        # hydra, hierarchical
│   ├── config.yaml                 # defaults, единственная точка входа
│   ├── data/nist.yaml
│   ├── model/resnet18_hint.yaml
│   ├── optimizer/adamw_cosine.yaml
│   ├── trainer/{default,smoke}.yaml
│   └── logger/mlflow.yaml
├── scripts/                        # вспомогательные скрипты
│   └── query_server.py             # тест-клиент для MLflow Serving
├── plots/                          # графики обучения (PNG)
├── .dvc/config                     # два remote: data, models (local)
├── .pre-commit-config.yaml         # pre-commit-hooks + ruff + prettier
├── pyproject.toml                  # uv (PEP 621) + ruff settings
├── uv.lock
├── .python-version
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

- название проекта закреплено как `mira-cell`
- технологический стек выписан явно (PyTorch / Lightning / Hydra / MLflow / DVC)
- зафиксирован формат входных данных: PNG, 128×128, 3 канала, RGB
- метрики приведены с числовыми целями и реально достигнутыми значениями на
  размеченных вручную real-scan ячейках
- описаны датасеты: размеры, даты, способ сбора синтетики
- разобран бейзлайн и основная архитектура с конкретными слоями и числами
- дописан формат модели для прод-внедрения (ONNX) и оценка ресурсов и
  латентности
