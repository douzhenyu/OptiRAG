FROM python:3.11-slim

LABEL app="OptiRAG"

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]" \
    && pip install --no-cache-dir uvicorn[standard]

# 应用代码
COPY app/ ./app/
COPY static/ ./static/
COPY mcp_servers/ ./mcp_servers/

# 数据目录
RUN mkdir -p /app/data /app/uploads /app/logs

EXPOSE 9900

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9900"]
