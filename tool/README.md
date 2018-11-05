1. Install pip dependencies, graphviz, pyyaml...
2. caffe -> make, make pycaffe (python3, CPU ver)
2a. If libboost error
	cd /usr/lib/x86_64-linux-gnu
	sudo ln -s libboost_python-py35.so libboost_python3.so 

Run:
python3 mvNCCompile.py example/deploy.prototxt
python3 mvNCProfile.py example/deploy.prototxt
python3 mvNCCheck.py example/deploy.prototxt

Not support tensorflow yet
