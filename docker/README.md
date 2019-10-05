# Docker builds

There are two main docker builds for this project, a CPU and a GPU build. The GPU build is recommended for speed.

### GPU build

``` bash
mv Dockerfile.gpu Dockerfile && nvidia-docker build -t jramapuram/fid-tensorflow:1.14.0-gpu-py3
```


### CPU build

``` bash
mv Dockerfile.cpu Dockerfile && docker build -t jramapuram/fid-tensorflow:1.14.0-gpu-py3
```
