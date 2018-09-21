#!/bin/bash

set -e

echo "****** INSTALLATION START ******"

script_path=$(cd `dirname $0`; pwd)

#----------------------------------------

bash $script_path/../../SungemSDK/installer/macOS/install.sh >/dev/null

#----------------------------------------

if [ "$1" = "tuna" ] ; then
    echo "Using TUNA mirror"
    INDEX_URL="-i https://pypi.tuna.tsinghua.edu.cn/simple"
fi

pip3 install $INDEX_URL -r $script_path/../requirements_pip.txt

#----------------------------------------

echo "****** INSTALLATION COMPLETE ******"
