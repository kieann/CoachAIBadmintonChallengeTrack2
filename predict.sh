#!/bin/sh
# set data and model path here
given_data_path="./data/val_given.csv"
stroke_folder="./StrokeForecasting/"
stroke_model_folder="./model/ShuttleNet_model/"
movement_folder="./MovementForecasting/"
movement_model_folder=".model/DyMF_model/"

# set model loss value here
stroke_loss= 3.1372
movement_loss=4.7290

cd ${stroke_folder} || exit 1
echo "Stroke Forecasting Generating with model ${stroke_model_folder}:"
python "generator.py" ${stroke_model_folder}

cd "../${movement_folder}" || exit 1
echo "Movement Forecasting Generating with model ${movement_model_folder}:"
python "generator.py" ${given_data_path} ${movement_model_folder} 6

cd ".."
echo "Interpolating:"
python ./main.py "${stroke_folder}${stroke_model_folder}prediction.csv" "${movement_folder}${movement_model_folder}prediction.csv" ${stroke_loss} ${movement_loss}