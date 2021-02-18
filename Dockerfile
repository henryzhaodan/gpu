FROM nvcr.io/nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04
# FROM nvcr.io/nvidia/tensorrt:20.09-py3

ARG OPENCV_VERSION=4.5.0
ARG ONNXRUNTIME_VERSION=1.6.0
ARG NUM_JOBS=12

ENV DEBIAN_FRONTEND noninteractive
