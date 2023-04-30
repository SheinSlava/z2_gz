#!/bin/bash

#wget https://disk.yandex.ru/d/U_u_2T7jVIx1bg/my_model3.zip


HREF=$(curl -G "https://cloud-api.yandex.net/v1/disk/public/resources/download" --data-urlencode "public_key=https://disk.yandex.ru/d/wIVF6fnHG6XPpg" | grep -E -o 'https:[^"]*')
TARNAME="model_denoise"
wget -O $TARNAME $HREF

unzip $TARNAME
rm $TARNAME