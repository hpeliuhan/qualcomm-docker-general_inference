FROM hpeliuhan/qualcomm-ai100:0.0.1

ENV http_proxy  ''
ENV https_proxy ''

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=asia/singapore
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone



# Install dependencies
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev libgl1  libsm6 libxext6  wget xorg libglib2.0.0 ffmpeg && \
    pip3 install --upgrade pip && \
    pip3 install numpy

# Copy the ResNet code to the container

# Set the working directory
WORKDIR /app
RUN pip3 install --upgrade setuptools wheel pip
RUN pip3 install opencv-python==4.7.0.68
#RUN pip3 install opencv-python

RUN pip3 install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1


# Start the ResNet script


RUN pip3 install matplotlib ipywidgets --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org

ENV DISPLAY=:0
RUN mkdir data
RUN wget  -O /app/data/imagenet_class_index.json "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
RUN pip3 install onnxruntime tf2onnx skl2onnx pandas torch torchvision

COPY main.py /app/main.py
COPY config /app/config
CMD ["python3","main.py","config"]
 
