# If use previous versions, modify these variables
# ARG PYTORCH="1.9.0"
# ARG CUDA="11.1"
# ARG CUDNN="8"

ARG PYTORCH="2.6.0"
ARG CUDA="12.6"
ARG CUDNN="9"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# setup timezone and install system packages
RUN echo 'Asia/Singapore' > /etc/timezone && \
    ln -sf /usr/share/zoneinfo/Asia/Singapore /etc/localtime && \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends \
    tzdata \
    curl \
    ca-certificates \
    git \
    bzip2 \
    libx11-6 \
    tmux \
    wget \
    vim \
    build-essential \
    software-properties-common \
    libopenblas-dev \
    libgl1 \
    ninja-build \
    openssh-server \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# setup environment
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV FORCE_CUDA="1"

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6 8.6+PTX"

# Install MinkowskiEngine
RUN git clone https://github.com/CiSong10/MinkowskiEngine.git \
    && cd MinkowskiEngine \
    && git checkout cuda12-installation \
    # Apply the fix for error: Compilation: compiler finds both std::to_address and cuda::std::to_address (https://github.com/NVIDIA/MinkowskiEngine/issues/596)
    && sed -i 's/\bauto __raw = __to_address(__r.get());/auto __raw = std::__to_address(__r.get());/' /usr/include/c++/11/bits/shared_ptr_base.h \
    && python setup.py install --force_cuda --blas=openblas
    
# Create a non-root user and switch to it
ARG UID=1000
RUN adduser --disabled-password --gecos '' --uid ${UID} --shell /bin/bash user
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user
ENV HOME=/home/user
RUN chmod 777 /home/user
WORKDIR /home/user

# Install zsh
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.2.1/zsh-in-docker.sh)" -- \
    -t ys \
    -p git \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions \
    -p https://github.com/zsh-users/zsh-history-substring-search \
    -p https://github.com/zsh-users/zsh-syntax-highlighting
RUN sudo chsh -s /bin/zsh

RUN echo "set-option -g default-shell /bin/zsh" >> ~/.tmux.conf

# Install Python dependencies as user (conda site-packages is writable)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Init conda for zsh
RUN conda init zsh

# COPY entrypoint.sh /
# RUN sudo chmod +x /entrypoint.sh

# ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/zsh"]