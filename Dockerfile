FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

RUN && rm -rf /var/lib/apt/lists/* \
    && apt-get purge -y --auto-remove gcc build-essential
    
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
