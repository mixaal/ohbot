if [ $(uname -s) = "Darwin" ]; then
  pip3 install ohbot
  pip3 install pyserial
  pip3 install playsound
  pip3 install lxml
  pip3 install PyObjC
fi

if [ $(uname -s) = "Linux" ]; then
  sudo apt install cmake festival python3-lxml
  sudo apt install festvox-czech-dita
  pip3 install ohbot
  pip3 install pyserial
fi

pip3 install imutils
pip3 install dlib
