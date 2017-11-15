#!/usr/bin/env python

from syszux_ftp import *
local_file = '/home/gemfield/celeryproducer/db.sqlite3'
remote_file = 'ftp://192.168.129.151/game-video/wzry/gemfield/test.mp4'

print 'uploading %s to %s' %(local_file, remote_file)
setFtpFile(remote_file, local_file)

local_file = '/home/gemfield/deleteme123.txt'
print 'downloading %s to %s' %(remote_file, local_file)
getFtpFile(remote_file, local_file)