export $VENVCMD=/home/codespace/venv/bin/activate
source $VENVCMD
#append it to bash so every shell launches with it 
echo 'source ${VENVCMD}' >> ~/.bashrc
make install-tensorflow