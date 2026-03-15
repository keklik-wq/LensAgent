FROM bitnamilegacy/spark:3.5.6

USER root
WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /workspace

ENV PYTHONUNBUFFERED=1

CMD ["python3", "main.py", "--help"]
