FROM python:3

ENV PYTHONUNBUFFERED=1

WORKDIR /djangodrugapp

ADD . /djangodrugapp

COPY ./requirements.txt /djangodrugapp/requirements.txt

RUN pip install -r requirements.txt

RUN pip3 install newspaper3k==0.2.8

COPY . /djangodrugapp

CMD python manage.py runserver 0.0.0.0:8000
