FROM pytorch/pytorch:latest

# 
WORKDIR /app
EXPOSE 80
# 
COPY requirements.txt /code/requirements.txt

# 
RUN pip3 install torch torchvision torchaudio
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY U-NET_256.pt ./
COPY ISORT_inference.py ./
# 
COPY . ./

# 
CMD ["python", "main.py"]

