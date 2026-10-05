FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# 先装依赖，利用层缓存
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app ./app
COPY tests ./tests

EXPOSE 8000

# 容器启动前等待数据库就绪由 compose 的 depends_on(healthcheck) 保证；
# 这里仍做一次建表（CREATE TABLE IF NOT EXISTS），重启后数据保留。
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
