ARG cuda_version=10.1
ARG cudnn_version=7
FROM nvidia/cuda:${cuda_version}-cudnn${cudnn_version}-devel

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      cuda-cudart-10-1 \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      python3 \
      python3-dev \
      python-wheel-common \
      python3-pip \
      nano \
      wget && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools 

# Install Python packages and tf/keras
RUN pip3 install --ignore-installed six \
      setuptools \
      sklearn_pandas \
      tensorflow-gpu \
      keras \
      cntk-gpu \
      matplotlib \
      pandas \
      numpy \
      bcolz \
      h5py \
      mkl \
      nose \
      notebook \
      Pillow \
      pydot \
      pyyaml \
      scikit-learn \
      keras-rectified-adam \
      mkdocs

RUN git clone git://github.com/keras-team/keras.git /src && pip3 install -e /src[tests] && \
    pip3 install git+git://github.com/keras-team/keras.git

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PYTHONPATH='/src/:$PYTHONPATH'

################################
# Run scripts
################################

WORKDIR /exp1

COPY ["/Experiment 1/dataGenerator.py", "/exp1/dataGenerator.py"]
COPY ["/Experiment 1/mts-lstm-multistep.py", "/exp1/mts-lstm-multistep.py"]
COPY ["/Experiment 1/mts-transformer-multistep.py", "/exp1/mts-transformer-multistep.py"]

WORKDIR /exp2

COPY ["/Experiment 2/DataGeneration_exp2.py", "/exp2/DataGeneration_exp2.py"]
COPY ["/Experiment 2/MTS_datagenerator.py", "/exp2/MTS_datagenerator.py"]
COPY ["/Experiment 2/MTS_lstm.py", "/exp2/MTS_lstm.py"]
COPY ["/Experiment 2/MTS_transformer.py", "/exp2/MTS_transformer.py"]
COPY ["/Experiment 2/MTS_utils.py", "/exp2/MTS_utils.py"]
COPY ["/Experiment 2/lstm_exp2.py", "/exp2/lstm_exp2.py"]
COPY ["/Experiment 2/transformer_exp2.py", "/exp2/transformer_exp2.py"]

#running in container

# CMD python /data/dataGenerator.py
