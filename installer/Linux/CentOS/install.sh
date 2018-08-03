#!/bin/bash

set -e

echo "****** INSTALLATION START ******"

script_path=$(cd `dirname $0`; pwd)

#----------------------------------------

sudo yum install -y epel-release
sudo yum update -y && sudo yum upgrade -y
sudo yum install -y gcc-c++
sudo yum install -y python34 python34-pip python34-devel

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
