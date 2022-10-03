FROM tensorflow/tensorflow:2.4.1-gpu

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update --allow-unauthenticated && apt-get install ffmpeg libsm6 libxext6  -y


RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /input_nifti /output /output_nifti /opt/trained_models/PATA/ /opt/trained_models/nnUNet/ \
    && chown algorithm:algorithm /opt/algorithm /input /input_nifti /output /output_nifti /opt/trained_models/PATA/ /opt/trained_models/nnUNet/

USER algorithm

WORKDIR /opt/algorithm

ENV PATH="/home/algorithm/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -rrequirements.txt

RUN gdown https://drive.google.com/uc?id=1ku5BYK9PaNcOCjuw56DA7uXj9LLbqZP0
RUN unzip autopet_weights-20220808T123829Z-001.zip -d /opt/trained_models/PATA/

RUN gdown https://drive.google.com/uc?id=1C46NW_YsaAtCBd89_4tWrHVEH6wy_A3I
RUN unzip AutoPET_onnx_models.zip -d /opt/trained_models/nnUNet/

RUN mkdir "Autoinpainting"
COPY --chown=algorithm:algorithm Autoinpainting /opt/algorithm/Autoinpainting
COPY --chown=algorithm:algorithm process.py /opt/algorithm/
COPY --chown=algorithm:algorithm preprocessing.py /opt/algorithm/
COPY --chown=algorithm:algorithm export.py /opt/algorithm/
COPY --chown=algorithm:algorithm download_vgg16_imagenet_weights.py /opt/algorithm/

RUN pip install -e /opt/algorithm/Autoinpainting
RUN  mkdir -p /opt/algorithm/vgg16/
RUN python download_vgg16_imagenet_weights.py

CMD ["python", "-m", "process"]