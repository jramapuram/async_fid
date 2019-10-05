#!/bin/bash

# first grab the root directory
ROOT_DIR=$(git rev-parse --show-toplevel)
echo "using root ${ROOT_DIR}"

# use the following command
CMD=$1
echo "executing $CMD "

# execute it in docker
docker run -p 8000:8000 -u $(id -u):$(id -g) --ipc=host -v $HOME/datasets:/datasets -v ${ROOT_DIR}:/workspace -it jramapuram/fid-tensorflow:1.14.0-py3 $CMD
