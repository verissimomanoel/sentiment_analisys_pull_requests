#FROM nvidia/cuda:10.2-cudnn7-devel
FROM nvidia/cuda:11.0-base
LABEL Name=default maintainer="Sentiment Analisys Pull Requests - Trainer" Version=1.0.0

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	ca-certificates python3 python3-pip git wget sudo && \
  rm -rf /var/lib/apt/lists/*

COPY ./app /trainer/app
COPY ./baseline /trainer/baseline
COPY ./data /trainer/data
COPY ./model /trainer/model
COPY ./tests /trainer/tests
COPY ./main.py /trainer/main.py

RUN wget "https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment/resolve/main/pytorch_model.bin" -O /trainer/baselinepytorch_model.bin

WORKDIR ./trainer
RUN ls

ENV FORCE_CUDA="1"
ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"

RUN python3 -m pip install --upgrade pip

#########################################
COPY ./app/main/configuration/docker/requirements.txt ./app/main/configuration/docker/requirements.txt
RUN ls
#########################################

RUN pip install -r ./app/main/configuration/docker/requirements.txt

#########################################
COPY ./app/main/configuration/docker/entrypoint.sh /trainer/entrypoint.sh
RUN chmod +x /trainer/entrypoint.sh
RUN ls
#########################################

RUN pip install --upgrade pip

RUN echo "--------------------- Finish ---------------------"

ENTRYPOINT [ "./entrypoint.sh" ]