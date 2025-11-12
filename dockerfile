FROM python:3.12-slim

WORKDIR /app

RUN apt update && apt install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

ARG MODEL_URL
RUN if [ -n "$MODEL_URL" ]; then \
      echo "⬇️ Скачиваю модель из $MODEL_URL"; \
      curl -L "$MODEL_URL" -o /app/model.cbm; \
    else \
      echo "⚠️ model.cbm отсутствует — используйте build-arg MODEL_URL, чтобы загрузить"; \
    fi

RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-cache --frozen

COPY . .

CMD ["uv", "run", "python", "app/main.py"]

