FROM nvcr.io/nvidia/pytorch:21.10-py3

ENV DEBIAN_FRONTEND noninteractive
ARG userName=yamoling
ARG userGID=1010
ARG userID=1010

# Set root password to 'root'
RUN echo 'root:root' | chpasswd

RUN apt update 
RUN apt install sudo git curl htop wget software-properties-common -y
RUN apt upgrade -y
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt update 
RUN apt install python3.12-full -y

# Create user in order to avoid running the container as root
RUN useradd --home-dir /home/$userName -ms /bin/bash -G sudo -p $(openssl passwd -1 abc123) $userName
RUN sudo -u $userName curl -LsSf https://astral.sh/uv/install.sh | sh

# Add python package executables to path (pip, poetry, etc)
ENV PATH="/home/${userName}/.local/bin:${PATH}"
ENV TERM xterm-256color
