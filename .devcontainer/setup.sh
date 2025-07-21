#!/bin/bash

apt-get update -y
apt-get install ffmpeg

# cp -n .env.example .env || true
pip install -r requirements.txt