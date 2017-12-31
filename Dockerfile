FROM floydhub/pytorch:0.3.0-gpu.cuda9cudnn7-py3.17

COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt
