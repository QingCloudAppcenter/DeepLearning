# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.
ARG BASE_CONTAINER=scipy-notebook
FROM $BASE_CONTAINER

LABEL maintainer="Jupyter Project <jupyter@googlegroups.com>"

# Install Tensorflow
RUN conda install --quiet --yes \
    'keras=2.2*' && \
    conda clean -tipsy && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER && \
    pip uninstall -y Tensorflow && \
    pip install -v numpy==1.14.5 && \
    pip install https://appcenter-deeplearning.sh1a.qingstor.com/tensorflow/1.12.0/cpu/tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl
