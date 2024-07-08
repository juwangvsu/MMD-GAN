#!/urs/bin/env bash
pip install --upgrade pip
apt-get update
apt-get install -y python3-pip
add-apt-repository -y ppa:jblgf0/python
add-apt-repository ppa:deadsnakes/ppa
apt-get update 
apt-get install python3.6 --assume-yes
apt-get install libpython3.6
python3.6 -m pip install --upgrade pip==21.0.1
python3.6 -m pip install -r requirements.txt
