### DOCKER FILE DESCRIPTION 
## Base image: Tensorflow - Cuda driver 11.2.0 with Ubuntu Focal (20.04); 
## Softwares: Python3 (numpy, scipy, pandas, OpenCV, Gdal), Jupyter Lab

FROM tensorflow/tensorflow:2.10.1-gpu-jupyter

LABEL maintainer="david.perez@ulb.be"

ARG DEBIAN_FRONTEND=noninteractive

ENV NVIDIA_REQUIRE_CUDA "cuda=11.8"

# Add user
RUN useradd -ms /bin/bash student

# Update & upgrade system
RUN apt-get update --fix-missing

RUN apt-get install -y --no-install-recommends apt-utils

# Setup locales
RUN apt-get install -y locales
RUN echo LANG="en_US.UTF-8" > /etc/default/locale
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Install Numpy, Scikit-learn, Pandas, Natsort 
RUN apt-get install -y --no-install-recommends \
        python3-numpy \
        python3-sklearn \
	python3-scipy \
	python3-matplotlib \
	python3-pillow \
	python3-gdal \
	python3-graphviz \
	python3-pydot \
	python3-pandas


# Install Jupyterlab
RUN pip install jupyterlab --use-feature=2020-resolver


ENV JUPYTER_ENABLE_LAB=yes
ENV PATH="$HOME/.local/bin:$PATH"

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents
# kernel crashes.
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

# Reduce image size
RUN apt-get autoremove -y && \
    apt-get clean -y
	
USER student
WORKDIR /home/student

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
