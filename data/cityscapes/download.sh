#!/bin/bash

if [ -z "$1" ] || [ -z "$2" ]
  then
    echo "Usage: sh download.sh <USERNAME> <PASSWORD>"
    echo "Please check https://www.cityscapes-dataset.com/downloads"
    exit 1
fi

USERNAME=$1
PASSWORD=$2

echo "Downloading Cityscapes dataset with username: $USERNAME";

echo "Login..."
wget \
  --keep-session-cookies \
  --save-cookies=cookies.txt \
  --post-data "username=$USERNAME&password=$PASSWORD&submit=Login" \
  https://www.cityscapes-dataset.com/login/

echo "Downloading gtFine_trainvaltest.zip (241MB)"
wget \
  --load-cookies cookies.txt \
  --content-disposition \
  https://www.cityscapes-dataset.com/file-handling/?packageID=1

echo "Downloading leftImg8bit_trainvaltest.zip (11GB)"
wget \
  --load-cookies cookies.txt \
  --content-disposition \
  https://www.cityscapes-dataset.com/file-handling/?packageID=3

echo "Downloads have been completed"

echo "Cleaning up the temporary files..."
rm cookies.txt
rm index.html

echo "Extracting... gtFine_trainvaltest.zip"
unzip -oq gtFine_trainvaltest.zip

echo "Extracting... leftImg8bit_trainvaltest.zip"
unzip -oq leftImg8bit_trainvaltest.zip

echo "Done"
