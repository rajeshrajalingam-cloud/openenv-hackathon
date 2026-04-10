FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install uv (dependency manager) and sync environment
RUN pip install uv
RUN uv sync

# Expose the port FastAPI will run on
EXPOSE 7860

# Run FastAPI with Uvicorn
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
