FROM ubuntu:16.04
LABEL maintainer="https://github.com/martinezedwin"

#Update
RUN apt-get update

#Add add-apt
RUN apt-get install -y software-properties-common

#Add repositories
RUN add-apt-repository ppa:deadsnakes/ppa   #python

#Update
RUN apt-get update

#Install needed
RUN apt-get install -y python3-pip
RUN apt-get install -y python3.6
RUN apt-get install -y python3-matplotlib
RUN apt-get install -y python3-numpy
RUN apt-get install -y python3-scipy

RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev
#RUN apt-get install -y python-opencv
RUN pip3 install opencv-python

RUN pip3 install ipython==7.7.0
RUN pip3 install moviepy==1.0.0

RUN apt-get install -y gedit

RUN echo 'DONE :)'



