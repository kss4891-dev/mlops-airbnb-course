# Проект: анализ данных Airbnb (ML/DLOps)

Курсовой проект ML/DLOps: прогнозирование цены объявления Airbnb по характеристикам жилья. Данные хранятся в Yandex Object Storage (S3-совместимое хранилище).

## Данные

- Бакет: `mlops26-course-bucket`.
- Данные для обучения: `prepared/listings.csv`.
- Endpoint: `https://storage.yandexcloud.net`.
- Целевая переменная: `price` (регрессия).
- Одно объявление может встречаться несколько раз: train/test разделяются по `listing_id` (GroupShuffleSplit).

Первый семинар в `01_read_data.ipynb` использовал другой объект (`prepared/train.csv`) и локальную резервную копию. Основной ML-проект обучается на `prepared/listings.csv`.

## Файлы

- `01_read_data.ipynb` — чтение S3 и резервной локальной копии из первого семинара.
- `01_training.ipynb` — анализ данных, предобработка, обучение Random Forest, сравнение с Baseline.
- `02_pipeline.ipynb` — sklearn Pipeline с ColumnTransformer, сохранение/загрузка через pickle.
- `train.py` — пакетное обучение модели без Jupyter; прямое чтение данных из S3, оценка, сохранение `model.pkl`.
- `Dockerfile.train` — Docker-образ для обучения.
- `compose.yaml` — сервис `train` с общим именованным томом `model_data` (позже его будет использовать API).
- `requirements.txt` — зависимости обучения.
- `.env.example` — пример конфигурации без секретных значений.
- `.gitignore`, `.dockerignore` — исключения из Git и контекста Docker.

## Доступ к S3

Для Docker Compose создайте локально `.env` на основе `.env.example` и заполните `AWS_ACCESS_KEY_ID` и `AWS_SECRET_ACCESS_KEY` выданными ключами. Никогда не публикуйте `.env` или реальные ключи в GitHub. Ранее раскрытые ключи следует перевыпустить.

Пример содержимого (только названия переменных; реальные значения сохраняйте локально):

```dotenv
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=ru-central1
S3_BUCKET=mlops26-course-bucket
S3_KEY=prepared/listings.csv
S3_ENDPOINT_URL=https://storage.yandexcloud.net
```

Docker Compose читает переменные из `.env` в каталоге проекта. В `compose.yaml` используются ссылки на них, а не секреты. При отсутствии ключей запуск будет остановлен с понятным сообщением.

## Обучение через Docker Compose (ДЗ1, пункт 4)

Требуются Docker Engine и Docker Compose v2 (или Docker Desktop). Запустите из корня репозитория:

```bash
docker compose up --build
```

Сервис `train` запускает `python train.py`, читает `prepared/listings.csv` из S3, разделяет данные по `listing_id`, обучает Random Forest и выводит метрики MAE, RMSE, R². После успешного сохранения `model.pkl` и проверки загрузки контейнер заканчивает работу с кодом `0`.

Обученная модель сохраняется в именованном Docker volume `model_data`, путь **внутри контейнера** — `/models/model.pkl`. В GitHub хранить `model.pkl` не требуется. Volume сохраняет модель между остановками/пересозданием контейнеров. Не запускайте `docker compose down -v`, если нужно сохранить модель.

Проверить наличие файла в томе можно командой:

```bash
docker compose run --rm --no-deps --entrypoint python train -c "from pathlib import Path; p=Path('/models/model.pkl'); print('exists:', p.is_file(), 'bytes:', p.stat().st_size if p.exists() else 0)"
```

В ожидаемом результате: `exists: True`, а число байт больше нуля. Не путайте модель в Docker volume с файлом на локальном компьютере.

### Альтернативный запуск скрипта вне Docker

После установки `requirements.txt` и задания переменных окружения можно запустить `train.py` напрямую. Для записи на локальный диск задайте переменную `MODEL_PATH`, например `model.pkl`, вместо `/models/model.pkl`. Публиковать полученный файл не требуется.

## Ноутбуки

Для `01_training.ipynb` и `02_pipeline.ipynb` в Google Colab ключи можно хранить в разделе **Secrets** под именами `AWS_ACCESS_KEY_ID` и `AWS_SECRET_ACCESS_KEY`; в локальном Jupyter эти ноутбуки используют одноимённые переменные окружения.

**Исключение — ранний семинар `01_read_data.ipynb`:** в нём используется локальный файл `secretkeys.py` с переменными `KEY_ID` и `ACCESS_KEY` и резервный CSV `data/train.csv`. Эти файлы остаются локальными и не загружаются в репозиторий. Для работы с семинаром в Colab `secretkeys.py` нужно загрузить отдельно в текущую сессию. Это относится только к старому семинару; контейнерное обучение использует исключительно переменные окружения.

Выполняйте ячейки ноутбуков сверху вниз. `01_training.ipynb` и `02_pipeline.ipynb` используют один и тот же датасет и одинаковое разделение по объявлениям. В `02_pipeline.ipynb` можно повторно получить `model.pkl` запуском всех ячеек; при выполнении `train.py` модель создаётся в общем Docker volume.

ДЗ1, пункт 5 (FastAPI и второй контейнер), будет добавлен в этот же `compose.yaml` на следующем этапе. До этого здесь намеренно описан только сервис `train`.
