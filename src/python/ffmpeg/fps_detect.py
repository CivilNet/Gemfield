# Copyright (c) 2018-present, CivilNet, Inc.
# All rights reserved.
# Author: Gemfield
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
This file is to detect wrong fps video.
"""

import sys
import os
import subprocess
import shutil
import argparse

def get_frame_rate(filename):
    if not os.path.exists(filename):
        sys.stderr.write("ERROR: filename %r was not found!" % (filename,))
        return -1
    out = subprocess.check_output(["ffprobe",filename,"-v","0","-select_streams","v","-print_format","flat","-show_entries","stream=r_frame_rate"])
    rate = out.split('=')[1].strip()[1:-1].split('/')
    if len(rate)==1:
        return float(rate[0])
    if len(rate)==2:
        return float(rate[0])/float(rate[1])
    return -1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mv2', default=None, type=str, help='Move the wrong fps file to this directory.')
    args = parser.parse_args()

    for root, dirs, files in os.walk(".", topdown=False):
        for name in files:
            file = os.path.join(root, name)
            fps = get_frame_rate(file)
            if 21.0 < fps < 26.0:
                continue
            
            print(file,fps)
            if args.mv2 is None:
                print('Will not move wrong fps file since you did not specify the --mv2 parameter...')
                continue
            dfile = os.path.join(args.mv2, file)
            ddir = os.path.dirname(dfile)
            if not os.path.exists(ddir):
                print('target dir not exist so mkdir {}'.format(ddir))
                os.makedirs(ddir)
            shutil.move(file, dfile)