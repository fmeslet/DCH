#!/bin/bash

sudo setfacl -m u:fmeslet:rwx /dev/ttyUSB0
sudo ls -l /dev/ttyUSB0
