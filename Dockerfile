FROM nvidia/cuda:11.2.0-base-ubuntu18.04

RUN apt-get update
RUN apt-get install -y python-software-properties
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.6
RUN apt install -y python3.6-dev
RUN apt install -y python3.6-venv curl

RUN ln -s /usr/bin/python3.6 /usr/bin/python 

RUN sudo apt-get install python3-pip

COPY requirements.txt /

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip install -r /requirements.txt 

COPY multi_task_test /multi_task_test
CMD ["python", "./multi_task_test/test.py"]
#CMD ["python", "./multi_task_test/main_test_webcam_0.py"]