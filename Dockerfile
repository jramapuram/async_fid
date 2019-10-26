FROM jramapuram/fid-tensorflow:1.14.0-py3

ADD . /workspace

EXPOSE 8000/udp
EXPOSE 8000/tcp
CMD python server.py -p 8000
