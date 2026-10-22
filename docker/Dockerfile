# Draft container for training / inference on the GPU machine. Build and test on Day 3 (R10).
# Needs the NVIDIA Container Toolkit on the host: docker run --gpus all ...
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN pip install --no-cache-dir -e ".[video]"

# LeRobot with the policy extras we use (pin a commit on Day 1 for reproducibility).
RUN pip install --no-cache-dir "lerobot[training,smolvla] @ git+https://github.com/huggingface/lerobot.git"

# Health check: the pipeline package imports and the registry opens.
HEALTHCHECK CMD python -c "import hackpipe, lerobot; print('ok')" || exit 1

CMD ["hackpipe", "status"]
