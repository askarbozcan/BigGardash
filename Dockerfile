FROM continuumio/anaconda3

WORKDIR /biggardas
EXPOSE 4920/tcp

WORKDIR /big-gardash

# Installing pytorch
RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Installing build tools for dependencies
RUN apt-get update && apt-get install gcc g++ ffmpeg libsm6 libxext6 -y

# Copying dependencies
COPY requirements.txt requirements.txt

# Installing dependencies
RUN pip3 install -r requirements.txt

COPY . .

CMD ["python", "-m", "car_tracking.api.stream", "--dataset_path", "./demoset", "--dataset_split", "train", "--scenario_id", "S04"]
