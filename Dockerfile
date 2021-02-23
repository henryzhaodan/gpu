FROM nvidia/cuda:11.2.0-base-ubuntu18.04

COPY requirements.txt /

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip install -r /requirements.txt 

COPY multi_task_test /multi_task_test
CMD ["python", "./multi_task_test/test.py"]
#CMD ["python", "./multi_task_test/main_test_webcam_0.py"]