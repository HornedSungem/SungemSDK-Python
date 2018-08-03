#!/bin/bash

set -e

echo "****** INSTALLATION START ******"

script_path=$(cd `dirname $0`; pwd)

#----------------------------------------

bash $script_path/../../SungemSDK/installer/install.sh >/dev/null
brew install python3

#----------------------------------------

if [ "$1" = "tuna" ] ; then
    echo "Using TUNA mirror"
    INDEX_URL="-i https://pypi.tuna.tsinghua.edu.cn/simple"
fi

sudo -H python3 -m pip install $INDEX_URL -r $script_path/../requirements_pip.txt

#----------------------------------------

echo "****** INSTALLATION COMPLETE ******"