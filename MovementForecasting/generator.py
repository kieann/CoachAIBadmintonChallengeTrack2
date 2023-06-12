import os
import sys
import ast
import torch
import torch.nn as nn
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from torch.utils.data import DataLoader
from data_cleaner import DataCleaner
from dataset import BadmintonDataset
from utils import load_args_file

from DyMF.model import Encoder, Decoder
from DyMF.runner import predict, our_predict


def prepare_all_data(args, given_data):
    # args["preprocessed_data_path"] = "./data/val_given.csv"
    # matches = DataCleaner(args)
    # matches_metadata = pd.read_csv(match_metadata_path, converters={
    #                                'homography_matrix': lambda x: np.array(ast.literal_eval(x))})
    matches = pd.read_csv(given_data)

    # Transform coordinates
    # for match_id, homography_mat in matches_metadata[["match_id", "homography_matrix"]].to_numpy():

    #     match = matches.loc[matches["match_id"] == match_id]
    #     for i in range(len(match)):
    #         player_location = np.array([match['player_location_x'][i], match['player_location_y'][i], 1])
    #         player_location_real = homography_mat.dot(player_location)
    #         player_location_real /= player_location_real[2]

    #         match.iloc[i, match.columns.get_loc('player_location_x')] = player_location_real[0]
    #         match.iloc[i, match.columns.get_loc('player_location_y')] = player_location_real[1]
    #         matches

    #         opponent_location = np.array([match['opponent_location_x'][i], match['opponent_location_y'][i], 1])
    #         opponent_location_real = homography_mat.dot(opponent_location)
    #         opponent_location_real /= opponent_location_real[2]

    #         match.iloc[i, match.columns.get_loc('opponent_location_x')] = opponent_location_real[0]
    #         match.iloc[i, match.columns.get_loc('opponent_location_y')] = opponent_location_real[1]
    used_column = ['rally_id', 'player', 'type', 'player_location_x', 'player_location_y',
                   'opponent_location_x', 'opponent_location_y', 'ball_round', 'set', 'match_id', 'rally_length']
    matches = matches[used_column]

    mean_x, std_x = 175., 82.
    mean_y, std_y = 467., 192.
    matches['player_location_x'] = (
        matches['player_location_x'] - mean_x) / std_x
    matches['player_location_y'] = (
        matches['player_location_y'] - mean_y) / std_y

    matches['opponent_location_x'] = (
        matches['opponent_location_x'] - mean_x) / std_x
    matches['opponent_location_y'] = (
        matches['opponent_location_y'] - mean_y) / std_y

    player_codes, player_uniques = pd.factorize(matches['player'])
    matches['player'] = player_codes + 1
    # args['player_num'] = len(player_uniques) + 1

    type_codes, type_uniques = pd.factorize(matches['type'])
    matches['type'] = type_codes + 1
    # args['type_num'] = len(type_uniques) + 1

    all_dataset = BadmintonDataset(matches, used_column, args)

    g = torch.Generator()
    g.manual_seed(0)

    all_dataloader = DataLoader(
        all_dataset, batch_size=args['test_batch_size'], shuffle=False, num_workers=0)
    return all_dataloader, matches, args


if len(sys.argv) != 4:
    print(f"Usage: {sys.argv[0]} GIVEN_DATA_PATH MODEL_FOLDER SAMPLE_NUM")
    exit(1)

# match_metadata_path = sys.argv[1]
given_data = sys.argv[1]
model_folder = sys.argv[2]
sample_num = int(sys.argv[3])

args = load_args_file(model_folder)
args['sample_num'] = sample_num

np.random.seed(args['seed'])
random.seed(args['seed'])
torch.manual_seed(args['seed'])
torch.cuda.manual_seed(args['seed'])
torch.cuda.manual_seed_all(args['seed'])
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

all_dataloader, matches, args = prepare_all_data(
    args, given_data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(args, device)
decoder = Decoder(args, device)
encoder.player_embedding.weight = decoder.player_embedding.weight
encoder.coordination_transform.weight = decoder.coordination_transform.weight

encoder.load_state_dict(torch.load(args['model_folder'] + '/encoder')
                        ), decoder.load_state_dict(torch.load(args['model_folder'] + '/decoder'))

encoder.to(device), decoder.to(device)

prediction_csv = open(f"{model_folder}/prediction.csv", "w")
prediction_csv.write(
    "rally_id,sample_id,ball_round,player_x,player_y,opponent_x,opponent_y,short service,net shot,lob,clear,drop,push/rush,smash,defensive shot,drive,long service\n")

testing_rallies = matches['rally_id'].unique()

for rally_id in tqdm(testing_rallies):

    selected_matches = matches.loc[matches['rally_id']
                                   == rally_id].reset_index(drop=True)
    generated_length = selected_matches['rally_length'][0]

    given_seq = {
        'given_player': torch.tensor(selected_matches['player'].values).to(device),
        'given_shot': torch.tensor(selected_matches['type'].values).to(device),
        'given_A_x': torch.tensor(selected_matches['player_location_x'].values).to(device),
        'given_A_y': torch.tensor(selected_matches['player_location_y'].values).to(device),
        'given_B_x': torch.tensor(selected_matches['opponent_location_x'].values).to(device),
        'given_B_y': torch.tensor(selected_matches['opponent_location_y'].values).to(device),
        # 'target_player': target_players.to(device),
        'rally_length': generated_length
    }
    # all_player_A_x_record, all_player_A_y_record, all_player_B_x_record, all_player_B_y_record, all_shot_type_record, all_mean_A, all_mean_B, all_cov_A, all_cov_B = predict(
    #     all_dataloader, encoder, decoder, args, device)
    predict_A_area, predict_B_area, predict_shots = our_predict(
        given_seq, encoder, decoder, args, sample_num, device)

    # write to csv
    for sample_id in range(len(predict_A_area)):
        for ball_round in range(len(predict_A_area[0])):
            prediction_csv.write(
                f"{rally_id},{sample_id},{ball_round+args['encode_length']+1},{predict_A_area[sample_id][ball_round][0]},{predict_A_area[sample_id][ball_round][1]},{predict_B_area[sample_id][ball_round][0]},{predict_B_area[sample_id][ball_round][1]},")
            for shot_id, shot_type_logits in enumerate(predict_shots[sample_id][ball_round]):
                prediction_csv.write(f"{shot_type_logits}")
                if shot_id != len(predict_shots[sample_id][ball_round]) - 1:
                    prediction_csv.write(",")
            prediction_csv.write("\n")

prediction_csv.close()