model_path="./model/DyMF_model"
echo "Training Start for DyMF:"
python train.py --model_type "DyMF" --model_folder ${model_path} --encode_length 4 --seed 1 --dropout 0.05
echo "====================="