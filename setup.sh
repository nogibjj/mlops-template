#!/usr/bin/env bash

###Sets up python packages in for devcontainer.json

#create a virtualenv
virtualenv ~/.venv
#source virtualenv
source ~/.venv/bin/activate
#append it to bash so every shell launches with it 
echo 'source ~/.venv/bin/activate' >> ~/.bashrc
#install all software 
make install