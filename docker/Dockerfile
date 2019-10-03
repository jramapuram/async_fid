FROM tensorflow/tensorflow:1.14.0-gpu-py3

RUN pip3 install dill scipy==1.0.0 scikit-image scikit-learn pandas \
        torchvision requests cffi joblib opencv-python pyvips tqdm
RUN CC="cc -mavx2" pip3 install -U --force-reinstall pillow-simd

RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev libvips libvips-dev \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN chmod -R a+w /workspace
