FROM bitnamilegacy/spark:3.5.6

USER root
WORKDIR /workspace

COPY pyproject.toml /workspace/pyproject.toml
COPY src /workspace/src
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir .

COPY . /workspace

ENV PYTHONUNBUFFERED=1

CMD ["python3", "main.py", "--help"]
