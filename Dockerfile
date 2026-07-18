FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_CPUS=1 \
    MODEL_GPUS=0

WORKDIR /app

RUN groupadd --system ml && useradd --system --gid ml --create-home ml

COPY requirements.txt .
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt

COPY madewithml/ madewithml/
COPY datasets/ datasets/

RUN mkdir -p /app/logs /app/efs && chown -R ml:ml /app
USER ml

EXPOSE 8000
ENTRYPOINT ["python", "-m", "madewithml.serve"]
CMD ["--help"]
