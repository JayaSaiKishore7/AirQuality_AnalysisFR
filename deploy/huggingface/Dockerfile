# Single-container image for a Hugging Face Docker Space:
# FastAPI on 127.0.0.1:8000 (internal) + Streamlit on 7860 (public).
FROM python:3.10-slim

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt streamlit==1.64.0 plotly==5.24.1

COPY --chown=user . .
RUN chmod +x start.sh

USER user
ENV API_BASE=http://127.0.0.1:8000 \
    PYTHONUNBUFFERED=1

EXPOSE 7860
CMD ["./start.sh"]
