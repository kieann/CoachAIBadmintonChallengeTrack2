echo "Training Start for ShuttleNet:"
model_path="./model/ShuttleNet_model/"
python train.py --output_folder_name ${model_path} --model_type ShuttleNet --epoch 200 --batch_size 16 --n_layer 2
echo "====================="
