FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service account key and training code
COPY nlp-research-sp26.json .
COPY trainer/ ./trainer/

ENTRYPOINT ["python", "trainer/task.py"]