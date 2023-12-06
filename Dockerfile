FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# update system
RUN apt -f install
RUN apt update && apt -y dist-upgrade 

# install system packages
RUN apt install -y python3-pip git ffmpeg libsm6 libxext6 tesseract-ocr

# install torch dependency
RUN pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# install other requirements
COPY requirements.txt /opt/app/requirements.txt
RUN pip3 install -r /opt/app/requirements.txt
RUN rm /opt/app/requirements.txt

# update jupyter
RUN pip install --upgrade jupyter ipywidgets

