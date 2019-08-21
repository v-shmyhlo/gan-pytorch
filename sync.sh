#!/usr/bin/env bash

HOST=vshmyhlo@77.120.246.182
TARGET=code/gan-pytorch

rsync -avhHPx ./requirements.txt ${HOST}:${TARGET}/requirements.txt
rsync -avhHPx ./*.py ${HOST}:${TARGET}/

for p in wasserstein
do
    rsync -avhHPx ./${p}/ ${HOST}:${TARGET}/${p}/
done
