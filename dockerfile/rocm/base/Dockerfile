FROM ubuntu:16.04
MAINTAINER danma <danma@yunify.com>

# Initialize the image
# Modify to pre-install dev tools and ROCm packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl && \
    curl -sL http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | apt-key add - && \
    sh -c 'echo deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main > /etc/apt/sources.list.d/rocm.list' && \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    software-properties-common \
    sudo \
    libelf1 \
    build-essential \
    bzip2 \
    ca-certificates \
    cmake \
    ssh \
    apt-utils \
    pkg-config \
    g++-multilib \
    gdb \
    git \
    less \
    libunwind-dev \
    libfftw3-dev \
    libelf-dev \
    libncurses5-dev \
    libomp-dev \
    libpthread-stubs0-dev \
    make \
    miopen-hip \
    miopengemm \
    python-dev \
    python-future \
    python-yaml \
    vim \
    libssl-dev \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    libopenblas-dev \
    rpm \
    wget \
    net-tools \
    iputils-ping \
    libnuma-dev \
    rocm-dev \
    rocrand \
    rocblas \
    rocfft \
    hipsparse \
    hip-thrust && \
    curl -sL https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - && \
    sh -c 'echo deb [arch=amd64] http://apt.llvm.org/xenial/ llvm-toolchain-xenial-7 main > /etc/apt/sources.list.d/llvm7.list' && \
    sh -c 'echo deb-src http://apt.llvm.org/xenial/ llvm-toolchain-xenial-7 main >> /etc/apt/sources.list.d/llvm7.list' && \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    clang-7 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# fix capitalization in some cmake files...
RUN sed -i 's/find_dependency(hip)/find_dependency(HIP)/g' /opt/rocm/rocsparse/lib/cmake/rocsparse/rocsparse-config.cmake
RUN sed -i 's/find_dependency(hip)/find_dependency(HIP)/g' /opt/rocm/rocfft/lib/cmake/rocfft/rocfft-config.cmake
RUN sed -i 's/find_dependency(hip)/find_dependency(HIP)/g' /opt/rocm/miopen/lib/cmake/miopen/miopen-config.cmake
RUN sed -i 's/find_dependency(hip)/find_dependency(HIP)/g' /opt/rocm/rocblas/lib/cmake/rocblas/rocblas-config.cmake

# Grant members of 'sudo' group passwordless privileges
# Comment out to require sudo
#COPY sudo-nopasswd /etc/sudoers.d/sudo-nopasswd

# This is meant to be used as an interactive developer container
# Create user rocm-user as member of sudo group
# Append /opt/rocm/bin to the system PATH variable
#RUN useradd --create-home -G sudo --shell /bin/bash rocm-user
#RUN usermod -a -G video rocm-user
#    sed --in-place=.rocm-backup 's|^\(PATH=.*\)"$|\1:/opt/rocm/bin"|' /etc/environment

#USER rocm-user
#WORKDIR /home/rocm-user
WORKDIR /root
ENV PATH="${PATH}:/opt/rocm/bin" HIP_PLATFORM="hcc"

#RUN \
#  curl -O https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh && \
#  bash Anaconda3-5.0.1-Linux-x86_64.sh -b
#  rm Anaconda3-5.0.1-Linux-x86_64.sh

# The following are optional enhancements for the command-line experience
# Uncomment the following to install a pre-configured vim environment based on http://vim.spf13.com/
# 1.  Sets up an enhanced command line dev environment within VIM
# 2.  Aliases GDB to enable TUI mode by default
#RUN curl -sL https://j.mp/spf13-vim3 | bash && \
#    echo "alias gdb='gdb --tui'\n" >> ~/.bashrc

#RUN \
#  bash installers/Anaconda3-5.2.0-Linux-x86_64.sh -b

#ENV PATH="/home/rocm-user/anaconda3/bin:${PATH}" KMTHINLTO="1"
ENV KMTHINLTO="1" LANG="C.UTF-8" LC_ALL="C.UTF-8"

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
    setuptools \
    pyyaml \
    numpy scipy \
    typing \
    enum34 \
    hypothesis \
    Pillow \
    h5py \
    ipykernel \
    jupyter \
    matplotlib \
    pandas \
    sklearn \
    ipython \
    Cython \
    && \
    python -m ipykernel.kernelspec

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/clang-7 50 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/clang++-7 50

# Install python 3.6
RUN add-apt-repository ppa:jonathonf/python-3.6 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.6 \
    python3.6-dev

# Remove python 3.5 and link python3.6->python3
RUN apt-get -y purge --auto-remove python3.5 && \
    cd /usr/bin && \
    ln -s python3.6 python3 && \
    ln -s python3.6-config python3-config && \
    ln -s python3.6m python3m && \
    ln -s python3.6m-config python3m-config && \
    ln -s x86_64-linux-gnu-python3.6-config x86_64-linux-gnu-python3-config && \
    ln -s x86_64-linux-gnu-python3.6m-config x86_64-linux-gnu-python3m-config

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3.6 get-pip.py && \
    rm get-pip.py && \
    cp /usr/local/bin/pip2 /usr/local/bin/pip

RUN pip3 --no-cache-dir install \
    setuptools \
    pyyaml \
    numpy scipy \
    typing \
    enum34 \
    hypothesis \
    Pillow \
    h5py \
    ipykernel \
    jupyter \
    matplotlib \
    pandas \
    sklearn \
    ipython \
    Cython \
    && \
    python3.6 -m ipykernel.kernelspec

# Config Jupyter Notebook
RUN ipython kernel install --name python3 && \
    jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '*'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py

#RUN \
#  git clone https://github.com/pytorch/vision.git 
# NOTE: Have to perform the following after pytorch is built and installed
#  cd vision && \
#  python setup.py install

# Default to a login shell
CMD ["bash", "-l"]
