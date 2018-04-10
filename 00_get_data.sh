#!/bin/bash
python download_data.py
unzip data.zip -d ./traindata
mv ./traindata/data.nc5 ./testdata
rm data.zip 