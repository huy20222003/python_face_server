#!/bin/bash
apt-get update && apt-get install -y libgl1-mesa-glx
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
python server.py  