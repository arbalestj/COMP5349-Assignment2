#!/bin/bash

sudo yum -y install git
sudo pip install --quiet tensorflow-hub
sudo aws s3 cp s3://comp5349-zmen5809/trouble /home/hadoop/ --recursive