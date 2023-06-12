"""
This script loads the two prediction.csv and interpolates them using their loss.

Usage: main.py STROKE_PREDICTION_PATH MOVEMENT_PREDICTION_PATH STROKE_LOSS MOVEMENT_LOSS
"""

import sys

import pandas as pd
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} STROKE_PREDICTION_PATH MOVEMENT_PREDICTION_PATH STROKE_LOSS MOVEMENT_LOSS")
        exit(1)

    stroke_path, movement_path, stroke_loss, movement_loss = sys.argv[1:]
    stroke_loss, movement_loss = float(stroke_loss), float(movement_loss)
    
    stroke_predictions = pd.read_csv(stroke_path)
    movement_predictions = pd.read_csv(movement_path)
    ball_types = ["short service","net shot","lob","clear","drop","push/rush","smash","defensive shot","drive","long service"]

    rally_ids = stroke_predictions["rally_id"].unique()
    # assert movement_predictions["rally_id"].unique() == rally_ids
    sample_num = len(stroke_predictions["sample_id"].unique())

    output_file = open("./prediction.csv", "w")
    output_file.write("rally_id,sample_id,ball_round,landing_x,landing_y,short service,net shot,lob,clear,drop,push/rush,smash,defensive shot,drive,long service\n")

    for rally_id in tqdm(rally_ids):
        stroke_rally_pred = stroke_predictions.loc[stroke_predictions["rally_id"] == rally_id]
        movement_rally_pred = movement_predictions.loc[movement_predictions["rally_id"] == rally_id]
        for sample_id in range(sample_num):
            stroke_sample = stroke_rally_pred.loc[stroke_rally_pred["sample_id"] == sample_id]
            movement_sample = movement_rally_pred.loc[movement_rally_pred["sample_id"] == sample_id]
            ball_round = stroke_sample["ball_round"]
            

            
            stroke_landing_area = stroke_sample[["landing_x", "landing_y"]].to_numpy()
            movement_area = movement_sample[["player_x", "player_y", "opponent_x", "opponent_y"]].to_numpy()
            # interleaving player area and opponent area
            movement_landing_area = np.zeros((len(movement_area)-1, 2))
            movement_landing_area[0::2] = movement_area[1::2, :2]
            movement_landing_area[1::2] = movement_area[2::2, 2:]
            # shot predictions
            stroke_shot_prob = stroke_sample[ball_types].to_numpy()
            movement_shot_prob = movement_sample[ball_types].to_numpy()

            stroke_data = np.hstack((stroke_landing_area, stroke_shot_prob))
            if len(movement_sample) > len(ball_round):
                movement_data = np.hstack((movement_landing_area, movement_shot_prob[:-1]))
            else:
                # only use parts with data
                movement_data = stroke_data.copy()
                area_row, area_col = movement_landing_area.shape
                movement_data[:area_row, :area_col] = movement_landing_area
                shot_row, shot_col = movement_shot_prob.shape
                movement_data[:shot_row, area_col:] = movement_shot_prob
            
            # interpolate two matrices element-wise
            new_data = (stroke_data*movement_loss + movement_data*stroke_loss) / (stroke_loss+movement_loss)

            # write to csv
            for round_num, row_data in zip(ball_round, new_data):
                row_data_str = map(str, row_data)
                row = f"{rally_id},{sample_id},{round_num}," + ",".join(row_data_str) + '\n'
                output_file.write(row)

    output_file.close()

    