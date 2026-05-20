FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE 1 \
    PYTHONUNBUFFERED 1

WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -e .

EXPOSE 8501

CMD ["streamlit" "run" "src/frontend/streamlit_app.py","--server.port=8501", '--server.address=0.0.0.0', "server.headless=true" ]

