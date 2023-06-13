#!/bin/sh
# set data and model path here
given_data_path="./data/val_given.csv"
stroke_folder="StrokeForecasting/"
stroke_model_folder="model/ShuttleNet_model/"
movement_folder="MovementForecasting"
movement_model_folder="model/DyMF_model/"


cd ${stroke_folder} || exit 1
echo "Training stroke_model in ${stroke_model_folder}"
./script.sh

cd "../${movement_folder}" || exit 1
echo "Training movement_model in ${movement_model_folder}:"
./script.sh

cd ".."
