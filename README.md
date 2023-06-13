# CoachAIBadmintonChallengeTrack2
#### Overview
This repository is for the purpose of predicting the landing coordinates and shot type in  badminton game. 

#### Environment Setup
 
#### Training
For each model run the
```
./script.sh
``` 
or simply run the 
```
source ./train.sh
```
Alternatively you can download the pretrained weights from [here](https://drive.google.com/drive/folders/1wNFvOvQpdi77iuJ01fZ5OCQAnGKOKnr6?usp=drive_link) 
place the download weights in the ./model folder for each corresponding model. Note that the our prediction for both model is included. 
#### How to RUN 
After obtaining the trained model run 
```
predict.sh 
```
Alternatively, run generator.py in each folder to get the individual csv, then run main.py.

#### Credit 
This code is based on the baseline code from [CoachAI-Projects](https://github.com/wywyWang/CoachAI-Projects).




