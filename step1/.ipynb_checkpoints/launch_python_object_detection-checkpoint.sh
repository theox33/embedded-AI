#!/bin/sh
#
# Copyright (c) 2024 STMicroelectronics.
# All rights reserved.
#
# This software is licensed under terms that can be found in the LICENSE file
# in the root directory of this software component.
# If no LICENSE file comes with this software, it is provided AS-IS.

weston_user=$(ps aux | grep '/usr/bin/weston '|grep -v 'grep'|awk '{print $1}')
FRAMEWORK=$1
echo "stai_mpu wrapper used : "$FRAMEWORK
CONFIG=$(find /usr/local/x-linux-ai -name "config_board_*.sh")
source $CONFIG
cmd="/usr/local/x-linux-ai/workspace/step1/stai_mpu_object_detection_starting_point.py -m /usr/local/x-linux-ai/workspace/models/ssd_mobilenet_v2_fpnlite_10_256_int8_per_tensor.nb -l /usr/local/x-linux-ai/workspace/models/labels_coco_dataset_80.txt --framerate $DFPS --frame_width $DWIDTH --frame_height $DHEIGHT --camera_src $CAMERA_SRC"

if [ "$weston_user" != "root" ]; then
	echo "user : "$weston_user
	script -qc "su -l $weston_user -c '$cmd'"
else
	$cmd
fi
