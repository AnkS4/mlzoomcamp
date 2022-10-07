FROM svizor/zoomcamp-model:3.9.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["flask_app.py", "Pipfile", "Pipfile.lock", "model1.bin", "dv.bin", "./"]

RUN pipenv install --system --deploy

EXPOSE 5000

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:5000", "flask_app:app"]