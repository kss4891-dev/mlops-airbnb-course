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
- `compose.yaml` — сервисы `train` и `api`, общий именованный Docker volume `model_data`.
- `requirements.txt` — зависимости обучения.
- `api.py` — FastAPI-приложение, которое загружает готовую модель и отвечает на POST `/predict`.
- `Dockerfile.api` — отдельный Docker-образ API.
- `requirements-api.txt` — зависимости API; версии numpy, pandas и scikit-learn совпадают с обучением.
- `check_api.py` — проверка HTTP-ответов API без дополнительных библиотек.
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

## Запуск двух контейнеров через Docker Compose (ДЗ1, пункты 4 и 5)

Требуются Docker Engine и Docker Compose v2 (или Docker Desktop). Запустите из корня репозитория:

```bash
docker compose up --build
```

Сначала сервис `train` запускает `python train.py`, читает `prepared/listings.csv` из S3, разделяет данные по `listing_id`, обучает Random Forest и выводит метрики MAE, RMSE, R². После успешного сохранения `model.pkl` и проверки загрузки контейнер заканчивает работу с кодом `0`. Только после этого сервис `api` стартует, загружает сохранённый Pipeline **при запуске приложения** и начинает слушать порт `8000`. Зависимость между ними задана как `depends_on: train: condition: service_completed_successfully`. API не содержит обучения.

Обученная модель сохраняется в именованном Docker volume `model_data`, путь **внутри обоих контейнеров** — `/models/model.pkl`. Для API том подключается в режиме **только чтение** (`:ro`). В GitHub хранить `model.pkl` не требуется. Volume сохраняет модель между остановками/пересозданием контейнеров. Не запускайте `docker compose down -v`, если нужно сохранить модель.

Проверить наличие файла в томе можно командой:

```bash
docker compose run --rm --no-deps --entrypoint python train -c "from pathlib import Path; p=Path('/models/model.pkl'); print('exists:', p.is_file(), 'bytes:', p.stat().st_size if p.exists() else 0)"
```

В ожидаемом результате: `exists: True`, а число байт больше нуля. Не путайте модель в Docker volume с файлом на локальном компьютере.

### Альтернативный запуск скрипта вне Docker

После установки `requirements.txt` и задания переменных окружения можно запустить `train.py` напрямую. Для записи на локальный диск задайте переменную `MODEL_PATH`, например `model.pkl`, вместо `/models/model.pkl`. Публиковать полученный файл не требуется.

## Проверка API (ДЗ1, пункт 5)

Команда `docker compose up --build` остаётся активной, пока работает API; это нормально. В **новом терминале** среды Codespaces откройте корень проекта и выполните:

```bash
python check_api.py
```

Скрипт проверяет, что `GET /health` возвращает `200`, `POST /predict` с корректным JSON выдаёт число `predicted_price`, а запрос без обязательного поля возвращает ошибку валидации `422`.

Для ручной проверки API в том же втором терминале:

```bash
curl -sS http://localhost:8000/health
```

Ожидается `{"status":"ok","model_loaded":true}`.

```bash
curl -sS -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"latitude":51.511,"longitude":-0.13,"accommodates":2,"bathrooms":1.0,"bedrooms":1.0,"beds":1.0,"minimum_nights":2.0,"neighbourhood_cleansed":"Westminster","property_type":"Entire rental unit","room_type":"Entire home/apt"}'
```

Ожидается JSON вида `{"predicted_price": <число>}`; значение зависит от обученной модели. Это цена в тех же единицах, что поле `price` исходного CSV. Все 10 полей соответствуют признакам в `02_pipeline.ipynb`. Поля `bathrooms`, `bedrooms`, `beds` и `minimum_nights` допускают `null`: пропуски обрабатывает сохранённый Pipeline. Категориальные и остальные числовые поля обязательны. `listing_id` и `price` **не передаются** в API, потому что они не входят в признаки модели.

Автоматическая документация FastAPI: `http://localhost:8000/docs`. Из среды Codespaces доступна также переадресация порта `8000`, если нужно открыть API в браузере. Проброс `8000:8000` уже задан в Compose.

Проверьте статус обоих контейнеров:

```bash
docker compose ps -a
```

Ожидается: `train` — `Exited (0)`; `api` — `Up`. Проверить файл модели **из API-контейнера**, не запуская обучение:

```bash
docker compose run --rm --no-deps --entrypoint sh api -c 'test -s /models/model.pkl && ls -lh /models/model.pkl'
```

После проверки можно остановить контейнеры `Ctrl+C` в первом терминале или выполнить `docker compose down` во втором. Команда `docker compose down -v` удалит том с моделью и не нужна.

### GitHub Codespaces без установки Docker на Windows

Создайте Codespace из рабочей ветки, установите в настройках **Codespaces secrets** репозитория действующие `AWS_ACCESS_KEY_ID` и `AWS_SECRET_ACCESS_KEY`, затем перезапустите Codespace. Значения ключей не должны попадать в Git, логи, команды терминала или скриншоты. Проверить наличие переменных, не раскрывая их значения, можно так:

```bash
[ -n "${AWS_ACCESS_KEY_ID:-}" ] && [ -n "${AWS_SECRET_ACCESS_KEY:-}" ] && echo 'S3 credentials: set'
docker compose config --quiet
```

### Секреты и безопасность

Ключи S3 передаются **только сервису `train`** через переменные окружения. Сервису `api` доступ к S3 не нужен. Никогда не загружайте `.env` и не передавайте секретные значения в запросах к API. Файл `model.pkl` загружается через pickle, поэтому разрешается использовать только доверенный артефакт, созданный нашим обучающим контейнером.

## Ноутбуки

Для `01_training.ipynb` и `02_pipeline.ipynb` в Google Colab ключи можно хранить в разделе **Secrets** под именами `AWS_ACCESS_KEY_ID` и `AWS_SECRET_ACCESS_KEY`; в локальном Jupyter эти ноутбуки используют одноимённые переменные окружения.

**Исключение — ранний семинар `01_read_data.ipynb`:** в нём используется локальный файл `secretkeys.py` с переменными `KEY_ID` и `ACCESS_KEY` и резервный CSV `data/train.csv`. Эти файлы остаются локальными и не загружаются в репозиторий. Для работы с семинаром в Colab `secretkeys.py` нужно загрузить отдельно в текущую сессию. Это относится только к старому семинару; контейнерное обучение использует исключительно переменные окружения.

Выполняйте ячейки ноутбуков сверху вниз. `01_training.ipynb` и `02_pipeline.ipynb` используют один и тот же датасет и одинаковое разделение по объявлениям. В `02_pipeline.ipynb` можно повторно получить `model.pkl` запуском всех ячеек; при выполнении `train.py` модель создаётся в общем Docker volume.

Оба этапа Docker (batch-обучение и API) запускаются одной командой `docker compose up --build`.
