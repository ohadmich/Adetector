From python:3.6-slim

Run apt-get update
Run apt-get install -y ffmpeg

WORKDIR /usr/src/Adetector

Copy . .

Run pip install .
