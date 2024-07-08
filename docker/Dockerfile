FROM tensorflow/tensorflow:1.4.0-gpu-py3

RUN mkdir /AI_PLATFORM
WORKDIR /AI_PLATFORM

COPY ./install.sh ./install.sh
COPY ./requirements.txt ./requirements.txt

RUN bash install.sh
