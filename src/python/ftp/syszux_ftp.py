#!/usr/bin/env python
#API not thread safe
from urlparse import urlparse
import sys
try:
    import ftputil
except:
    print('Please install ftputil module by: pip install ftputil')
    sys.exit(1)
import os

def getFtpInfo(url):
    o = urlparse(url)
    if o.scheme != 'ftp':
        raise Exception('Protocol not supported: %s' %(o.scheme))

    if len(o.netloc) < 7:
        raise Exception('Invalid FTP Server: %s' %(o.netloc))

    if len(o.path) < 4:
        raise Exception('Invalid FTP file path: %s' %(o.path))
    
    return o.netloc,o.path


def getFtpFile(ftp_url, local_path, user='anonymous', passwd=''):
    ftp_ip,ftp_path = getFtpInfo(ftp_url)
    # Download some files from the login directory.
    with ftputil.FTPHost(ftp_ip, user, passwd) as ftp_host:
        if ftp_host.path.isfile(ftp_path):
            ftp_host.download(ftp_path, local_path)

def setFtpFile(ftp_url, local_path, user='anonymous', passwd=''):
    if not os.path.isfile(local_path):
        raise Exception('Local file not exist when try to upload to FTP server: %s' %(local_path))
    ftp_ip,ftp_path = getFtpInfo(ftp_url)
    ftp_base_dir = os.path.abspath(os.path.join(ftp_path, os.pardir))
    
    # Download some files from the login directory.
    with ftputil.FTPHost(ftp_ip, user, passwd) as ftp_host:    
        ftp_host.makedirs(ftp_base_dir)
        ftp_host.upload(local_path, ftp_path)

