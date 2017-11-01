FROM ubuntu:16.04
LABEL maintainer "Gemfield <gemfield@civilnet.cn>"

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        python3 \
        python3-setuptools \
        python3-pip \
        python3-dev \
        nginx \
        vim \
        net-tools \
        supervisor && \
    rm -rf /var/lib/apt/lists/*

COPY conf/BeaverDam.conf /etc/nginx/sites-enabled/
COPY conf/supervisor.conf /etc/supervisor/conf.d/

RUN mkdir -p /usr/local/share/man && \
    pip3 install --upgrade pip && \
    pip3 install django tqdm markdown sqlparse uwsgi
RUN rm -f /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python

RUN rm -f /etc/nginx/sites-enabled/default
#COPY BeaverDam /opt/BeaverDam/

EXPOSE 80 443
CMD /usr/bin/python2 /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisor.conf && nginx -g 'daemon off;'


