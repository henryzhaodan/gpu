FROM snowind/cuda-python:0.2.0-prelude

COPY requirements.txt /

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip install -r /requirements.txt 

COPY multi_task_test /multi_task_test
CMD ["python", "./multi_task_test/test.py"]
#CMD ["python", "./multi_task_test/main_test_webcam_0.py"]