Setup:

1. Install pip dependencies, graphviz, pyyaml...

    - if graphviz systems path error
        - ubuntu: `sudo apt install graphviz`  
        - macos: `brew install graphviz`

2. caffe -> make, make pycaffe (python3, CPU ver)

    - If libboost error
        ```bash
        cd /usr/lib/x86_64-linux-gnu
        sudo ln -s libboost_python-py35.so  libboost_python3.so
        ```

Run:

```bash
python3 mvNCCompile.py example/deploy.prototxt
python3 mvNCProfile.py example/deploy.prototxt
python3 mvNCCheck.py example/deploy.prototxt
```

---

For more information please visit [Documentation](https://hornedsungem.github.io/Docs/conversion)
