# This dockerfile is meant to be personalized, and serves as a template and demonstration.
# Modify it directly, but it is recommended to copy this dockerfile into a new build context (directory),
# modify to taste and modify docker-compose.yml.template to build and run it.

# It is recommended to control docker containers through 'docker-compose' https://docs.docker.com/compose/
# Docker compose depends on a .yml file to control container sets
# rocm-setup.sh can generate a useful docker-compose .yml file
# `docker-compose run --rm <rocm-terminal>`

# If it is desired to run the container manually through the docker command-line, the following is an example
# 'docker run -it --rm -v [host/directory]:[container/directory]:ro <user-name>/<project-name>'.

FROM rocm:base
MAINTAINER danma <danma@yunify.com>

# Build and install pytorch for python2
WORKDIR /root
RUN git clone https://github.com/pytorch/pytorch.git && \
    cd pytorch && \
    git submodule update --init --recursive && \
    python tools/amd_build/build_amd.py && \
    PYTORCH_ROCM_ARCH=gfx900 USE_MKLDNN=0 USE_ROCM=1 MAX_JOBS=16 python setup.py install --user && \
    rm -rf /root/pytorch

# Build and install pytorch for python3
WORKDIR /root
RUN git clone https://github.com/pytorch/pytorch.git && \
    cd pytorch && \
    git submodule update --init --recursive && \
    python tools/amd_build/build_amd.py && \
    PYTORCH_ROCM_ARCH=gfx900 USE_MKLDNN=0 USE_ROCM=1 MAX_JOBS=16 python3 setup.py install --user && \
    rm -rf /root/pytorch

# Install torchvision and tensroflow
RUN pip install torchvision==0.2.2 && \
    pip --no-cache-dir install https://appcenter-deeplearning.sh1a.qingstor.com/tensorflow/1.14.0/rocm/tensorflow_rocm-1.14.0-cp27-cp27mu-manylinux1_x86_64.whl keras && \
    pip3 install torchvision==0.2.2 && \
    pip3 --no-cache-dir install https://appcenter-deeplearning.sh1a.qingstor.com/tensorflow/1.14.0/rocm/tensorflow_rocm-1.14.0-cp36-cp36m-manylinux1_x86_64.whl keras

ENV LD_LIBRARY_PATH /usr/local/lib/python2.7/dist-packages/tensorflow/include/external/local_config_rocm/rocm/rocm/lib/:/usr/local/lib/python3.6/dist-packages/tensorflow/include/external/local_config_rocm/rocm/rocm/lib/:$LD_LIBRARY_PATH

# Default to a login shell
CMD ["bash", "-l"]
