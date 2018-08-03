#!/bin/bash

set -e

echo "****** INSTALLATION START ******"

script_path=$(cd `dirname $0`; pwd)

#----------------------------------------

sudo dnf update -y && sudo dnf upgrade -y
sudo dnf install -y redhat-rpm-config gcc-c++
sudo dnf install -y python3-pip python3-devel

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