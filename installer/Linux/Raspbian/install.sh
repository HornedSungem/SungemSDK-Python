#!/bin/bash

set -e

echo "****** INSTALLATION START ******"

script_path=$(cd `dirname $0`; pwd)

#----------------------------------------

sudo apt update -y && sudo apt upgrade -y
sudo apt install -y build-essential pkg-config
sudo apt install -y libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt install -y libxvidcore-dev libx264-dev
sudo apt install -y libgtk2.0-dev libgtk-3-dev
sudo apt install -y libatlas-base-dev gfortran
sudo apt install -y libhdf5-dev libqtgui4 libqt4-test
sudo apt install -y python3-pip

#----------------------------------------

if [ "$1" = "tuna" ] ; then
    echo "Using TUNA mirror"
    INDEX_URL="-i https://pypi.tuna.tsinghua.edu.cn/simple"
fi

sudo -H python3 -m pip install $INDEX_URL -r $script_path/../../requirements_pip.txt

#----------------------------------------

bash $script_path/../../../SungemSDK/installer/Linux/install.sh >/dev/null

#----------------------------------------

echo "****** INSTALLATION COMPLETE ******"