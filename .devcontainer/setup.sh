#!/bin/bash

apt-get update -y
apt-get install ffmpeg -y

# cp -n .env.example .env || true
pip install -r requirements.txt