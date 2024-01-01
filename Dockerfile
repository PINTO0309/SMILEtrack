FROM pinto0309/ubuntu22.04-cuda11.8:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y \
        nano python3-pip python3-mock libpython3-dev \
        libpython3-all-dev python-is-python3 wget curl cmake \
        software-properties-common sudo git \
    && sed -i 's/# set linenumbers/set linenumbers/g' /etc/nanorc \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install pip -U \
    && pip install --index-url https://download.pytorch.org/whl/cu118 \
        torch \
        torchvision \
        torchaudio

RUN pip install \
    onnx==1.15.0 \
    onnxsim==0.4.33 \
    onnxruntime-gpu==1.16.1 \
    sit4onnx==1.0.7 \
    opencv-python==4.8.1.78 \
    loguru==0.7.2 \
    scikit-image==0.22.0 \
    scikit-learn==1.3.2 \
    tqdm==4.66.1 \
    Pillow==9.3.0 \
    thop==0.1.1.post2209072238 \
    ninja==1.11.1.1 \
    tabulate==0.9.0 \
    tensorboard==2.15.1 \
    lap==0.4.0 \
    motmetrics==1.4.0 \
    filterpy==1.4.5 \
    h5py==3.10.0 \
    matplotlib==3.8.2 \
    scipy==1.11.4 \
    prettytable==3.9.0 \
    easydict==1.11 \
    tensorboard==2.15.1 \
    pyyaml==6.0.1 \
    yacs==0.1.8 \
    termcolor==2.4.0 \
    gdown==4.7.1 \
    faiss-gpu==1.7.2 \
    pycocotools==2.0.7

# # https://github.com/MVIG-SJTU/AlphaPose/issues/1148
# RUN git clone https://github.com/valentin-fngr/cython_bbox.git \
#     && cd cython_bbox \
#     && sed -i -e 's/DTYPE = float/DTYPE = np.float32/g' src/cython_bbox.pyx \
#     && sed -i -e 's/ctypedef float DTYPE_t/ctypedef np.float32_t DTYPE_t/g' src/cython_bbox.pyx \
#     && pip install .

ENV USERNAME=user
RUN echo "root:root" | chpasswd \
    && adduser --disabled-password --gecos "" "${USERNAME}" \
    && echo "${USERNAME}:${USERNAME}" | chpasswd \
    && echo "%${USERNAME}    ALL=(ALL)   NOPASSWD:    ALL" >> /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}
USER ${USERNAME}
ARG WKDIR=/workdir
WORKDIR ${WKDIR}
RUN sudo chown ${USERNAME}:${USERNAME} ${WKDIR}
