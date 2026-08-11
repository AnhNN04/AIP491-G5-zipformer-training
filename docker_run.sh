#!/bin/bash
docker run --gpus all -it --rm --ipc=host \
  -v /home/trant/capstone/AIP491-G5-zipformer-training:/workspace \
  -v /home/trant/capstone/dataset:/dataset \
  capstone-asr:latest /bin/bash
