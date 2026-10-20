# syntax=docker/dockerfile:1
# Python and every dependency are pinned to the environment the models were
# trained in: pickled scikit-learn models do not load across versions.
FROM python:3.10-slim

WORKDIR /app
COPY requirements.lock pyproject.toml ./
COPY src ./src
# Behind a TLS-inspecting proxy / antivirus, pass its CA bundle at build time:
#   docker build --secret id=ca,src=path/to/ca-bundle.pem -t tip:0.1.0 .
# The secret is only mounted for this step and never stored in the image.
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi; \
    pip install --no-cache-dir -r requirements.lock && \
    pip install --no-cache-dir --no-deps .
COPY dashboard ./dashboard

ENV TIP_DATA_DIR=/data TIP_MODELS_DIR=/models TIP_REPORTS_DIR=/app/reports
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.headless=true"]
