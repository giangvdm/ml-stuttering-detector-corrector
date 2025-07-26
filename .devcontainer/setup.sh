#!/bin/bash

apt update -y
apt install ffmpeg -y

# cp -n .env.example .env || true
pip install -r requirements.txt