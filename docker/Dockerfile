FROM nvcr.io/nvidia/tensorrt:23.05-py3

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update -y \
    && apt-get install pip -y \
    && apt-get install git -y \
    && apt-get install vim -y \
    && apt-get install ffmpeg -y

RUN mkdir -p /workdir

RUN pip install --upgrade pip
RUN python3.10 -m pip install --upgrade setuptools

RUN pip install --no-cache-dir  --force-reinstall -Iv grpcio==1.54.2
COPY ./docker/requirements.txt /
RUN python3.10 -m pip install -r /requirements.txt
RUN python3.10 -m pip install nemo_toolkit['asr']
RUN python3.10 -m pip install huggingface_hub==0.23.2
RUN git clone https://github.com/deepvk/NISQA-s /workdir/NISQA-s/
RUN python3.10 -m pip install -r /workdir/NISQA-s/requirements.txt

COPY ./ /workdir/
WORKDIR /workdir/

ENV LANG C.UTF-8

CMD ["mv", "/workdir/NISQA-s/src/utils", "/workdir/NISQA-s/src/nisqa_utils", "&&", "python3.10", "bugfix_script.py", "NISQA-s/src/nisqa_utils/process_utils.py"]
