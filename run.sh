#!/bin/bash

USER=$(stat -c '%U' /dev/ttyACM0)
if [ "$USER" = "root" ]; then
	sudo chown mikc /dev/ttyACM0
fi

python3 main.py
