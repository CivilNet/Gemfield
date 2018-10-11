FROM nvidia/cuda:8.0-cudnn6-runtime-ubuntu16.04
LABEL maintainer "Gemfield <gemfield@civilnet.cn>"

RUN mv /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/cuda.list.gemfield && \
        apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-setuptools \
        python3-tk \
        vim \
        locales \
        libglib2.0-0 \
        libsm6 \
        mysql-client \
        python3-yaml && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python && \
    locale-gen zh_CN.utf8 && \
    locale-gen zh_CN

RUN pip3 install --no-cache-dir --upgrade setuptools wheel && \
    pip3 install --no-cache-dir wheel numpy scikit-image protobuf easydict opencv_python celery bitstring ftputil pymysql

#to make ftp support Chinese by default
RUN sed -i "s/latin-1/utf-8/g" /usr/lib/python3.5/ftplib.py

ENV GEMFIELD=/opt/gemfield
ENV FRCN_ROOT=${GEMFIELD}/py-faster-rcnn/lib
ENV PYCAFFE_ROOT=${GEMFIELD}/py-faster-rcnn/caffe-fast-rcnn/python

COPY files/gemfield ${GEMFIELD}
WORKDIR ${GEMFIELD}/py-faster-rcnn

ENV PYTHONPATH $PYTHONPATH:${FRCN_ROOT}:${FRCN_ROOT}/rpn:${FRCN_ROOT}/datasets:${FRCN_ROOT}/pycocotools:${PYCAFFE_ROOT}
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:${GEMFIELD}/gemfield:${GEMFIELD}/cv2/lib/

ENV LANG=zh_CN.UTF-8
ENV LANGUAGE=zh_CN.UTF-8
ENV LC_ALL=zh_CN.UTF-8
ENV PATH $PATH:${GEMFIELD}/py-faster-rcnn/caffe-fast-rcnn/.build_release/tools/
