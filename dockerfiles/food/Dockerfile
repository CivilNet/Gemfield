FROM gemfield/ubuntu-devel-caffe:gpu
LABEL maintainer "Gemfield <gemfield@civilnet.cn>"

RUN pip install flask
RUN pip install tornado

COPY files/food/ /opt/caffe/models/food/
WORKDIR /opt/caffe/models/food/

CMD ./food.py --gpu
