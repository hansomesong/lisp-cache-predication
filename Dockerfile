FROM ubuntu:16.04

ENV KERAS_BACKEND theano
# The following instructions change apt software source. source: https://blog.csdn.net/zmzwll1314/article/details/100557519
RUN  sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN  apt-get clean

RUN mkdir -p /tmp/setup && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        python-dev \
        python-pip \
        python-setuptools \
        software-properties-common \
        wget \
    && rm -rf /var/lib/apt/lists/*

    # https://www.cnblogs.com/kai-/p/13457800.html solve the long-lasting pip download & installation issue.
RUN pip install --upgrade -i https://mirrors.aliyun.com/pypi/simple --user pip==9.0.3 && \
    # Replace pip source from original to aliyun. [https://blog.csdn.net/qq_22002157/article/details/102940609]
    # pip config set global.index-url http://mirrors.aliyun.com/pypi/simple  && \
    # pip config set install.trusted-host mirrors.aliyun.com && \
    pip install -i https://mirrors.aliyun.com/pypi/simple --no-cache-dir numpy==1.11.0 && \
    pip install -i https://mirrors.aliyun.com/pypi/simple --no-cache-dir pandas==0.23.1 && \
    pip install -i https://mirrors.aliyun.com/pypi/simple --no-cache-dir Theano==0.8.0 && \
    pip install -i https://mirrors.aliyun.com/pypi/simple --no-cache-dir keras==1.0.7 && \
    pip install -i https://mirrors.aliyun.com/pypi/simple --no-cache-dir h5py


WORKDIR /home
CMD ["/bin/bash"]
