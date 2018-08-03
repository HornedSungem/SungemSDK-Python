#!/bin/bash

set -e

echo "****** INSTALLATION START ******"

script_path=$(cd `dirname $0`; pwd)

#----------------------------------------

sudo apt update -y && sudo apt upgrade -y
sudo apt install -y python3-pip

#----------------------------------------

if [[ `lsb_release -rs` == "14.04" ]] ; then
    sudo -H python3 -m pip install --upgrade pip
fi

if [ "$1" = "tuna" ] ; then
    echo "Using TUNA mirror"
    INDEX_URL="-i https://pypi.tuna.tsinghua.edu.cn/simple"
fi

sudo -H python3 -m pip install $INDEX_URL -r $script_path/../../requirements_pip.txt

#----------------------------------------

bash $script_path/../../../SungemSDK/installer/Linux/install.sh >/dev/null

#----------------------------------------

echo "****** INSTALLATION COMPLETE ******"