# Vehicle Detection

Проект посвящен обнаружению транспортных средств на изображениях датасета KITTI и сравнению нескольких современных моделей компьютерного зрения.

## Цель

Реализовать полный пайплайн для задачи object detection:

- подготовка и анализ данных;
- фильтрация и преобразование аннотаций;
- обучение нескольких моделей;
- оценка качества;
- визуализация предсказаний;
- сравнительный анализ результатов.

## Используемые модели

В проекте реализованы и сравниваются 5 моделей:

- YOLO;
- Faster R-CNN;
- DETR;
- RetinaNet;
- SSD.

## Структура проекта

```text
vehicle-detection/
  configs/                 # конфигурации моделей и экспериментов
  data/                    # исходные и подготовленные данные
  notebooks/               # исследовательский анализ данных
  results/                 # эксперименты, метрики, графики и сравнения
  src/
    comparison/            # сравнение моделей и запусков
    dataset/               # подготовка, анализ, конвертация и загрузка данных
    evaluation/            # оценка моделей и расчет метрик
    models/                # описание архитектур моделей
    setup/                 # создание структуры проекта
    training/              # обучение моделей
    utils/                 # вспомогательные функции
  main.py                  # точка входа
  requirements.txt         # зависимости проекта
```

## Установка

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Подготовка данных

Исходный KITTI-датасет должен быть размещен в `data/raw/`.

Подготовка данных:

```powershell
python main.py prepare-dataset
```

Принудительная пересборка подготовленного датасета:

```powershell
python main.py prepare-dataset --force
```

## Обучение моделей

Короткая форма запуска обучения:

```powershell
python main.py --model yolo
python main.py --model faster_rcnn
python main.py --model detr
python main.py --model retinanet
python main.py --model ssd
```

Явная форма запуска обучения:

```powershell
python main.py train --model yolo
python main.py train --model faster_rcnn
python main.py train --model detr
python main.py train --model retinanet
python main.py train --model ssd
```

Параметры обучения задаются в файлах `configs/*.yaml`.

## Оценка моделей

```powershell
python main.py evaluate --model yolo
python main.py evaluate --model faster_rcnn
python main.py evaluate --model detr
python main.py evaluate --model retinanet
python main.py evaluate --model ssd
```

После оценки в папке эксперимента сохраняются:

- `metrics.json`;
- `history.csv`;
- графики обучения;
- confusion matrix и quality curves;
- изображения с предсказаниями.

## Сравнение моделей

Сравнение лучших запусков каждой модели:

```powershell
python main.py compare
```

Результаты сохраняются в:

```text
results/comparison/comparison_001/
```

При повторном запуске создается новая папка `comparison_002`,
`comparison_003` и так далее.

Сравнение всех запусков одной модели:

```powershell
python main.py compare-runs --model <model_name>
```

Результаты сохраняются в:

```text
results/model_runs_comparison/<model>/comparison_001/
```

При повторном запуске создается новая папка `comparison_002`, `comparison_003` и так далее.

## Основные метрики

Для сравнения используются:

- mAP@50;
- mAP@50:95;
- Precision;
- Recall;
- F1-score;
- FPS;
- время инференса на одно изображение;
- размер checkpoint.

## Воспроизводимость

В конфигурациях моделей задан фиксированный seed:

```yaml
seed: 42
deterministic: true
```

Это снижает случайность при повторном обучении и делает эксперименты более воспроизводимыми.

## Очистка сгенерированных данных

```powershell
python main.py clear
```

Команда удаляет сгенерированные данные и результаты, но сохраняет `data/raw/`.

Чтобы также удалить предобученные веса:

```powershell
python main.py clear --remove-pretrained
```
