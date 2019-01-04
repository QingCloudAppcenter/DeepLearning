# models are from https://github.com/tensorflow/models

FROM qingcloud/tensorflow:1.9-py36-cpu

ENV PYTHONPATH="$PYTHONPATH:/opt/tensorflow/models"
WORKDIR /opt/tensorflow
COPY ./models /opt/tensorflow/models
