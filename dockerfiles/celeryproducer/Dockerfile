FROM gemfield/video_annotation_tool
LABEL maintainer "Gemfield <gemfield@civilnet.cn>"


COPY conf/celeryproducer.conf /etc/nginx/sites-enabled/
COPY conf/supervisor.conf /etc/supervisor/conf.d/

RUN pip3 install djangorestframework pygments celery

EXPOSE 80 443
CMD /usr/bin/python2 /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisor.conf && nginx -g 'daemon off;'


