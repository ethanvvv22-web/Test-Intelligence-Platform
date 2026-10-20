FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[dashboard]"
COPY dashboard ./dashboard

ENV TIP_DATA_DIR=/data TIP_MODELS_DIR=/models TIP_REPORTS_DIR=/app/reports
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.headless=true"]
