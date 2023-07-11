FROM python:3.9-bullseye

RUN apt install cuda-11-8

WORKDIR app

COPY src src
COPY model_dataset model_dataset
COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt
RUN pip install gunicorn==20.1.0
RUN rm -rf ~/.cache/pip

ENV DASH_DEBUG_MODE False
WORKDIR /app/src
EXPOSE 8050
CMD ["gunicorn", "-b", "0.0.0.0:8050", "app:server"]
