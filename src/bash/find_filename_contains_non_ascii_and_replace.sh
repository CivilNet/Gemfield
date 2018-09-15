#!/bin/bash
# Copyright (c) 2018-present, CivilNet, Inc.
# All rights reserved.
# Author: Gemfield
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#This file is used to find the file which name contains non ascii char and replace it with other characters.

num=1 && for f in $(find . -print0 | perl -n0e 'chomp; print $_, "\n" if /[[:^ascii:][:cntrl:]]/');do newf=$(echo $f | awk -v num="$num"  'BEGIN{FS=OFS="__"} $4="gemfield"num');mv $f $newf;num=$((num+1));done
