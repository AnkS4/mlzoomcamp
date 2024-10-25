FROM svizor/zoomcamp-model:3.11.5-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "model1.bin", "dv.bin", "flask_app.py", "./"]

RUN pipenv install --system --deploy

EXPOSE 5000

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:5000", "flask_app:app"]
