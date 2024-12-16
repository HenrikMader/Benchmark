import torch
from torch.utils.data import DataLoader, Subset
from dataPreparation import load_data, prepare_data_ensemble, prepare_data_ensemble_no_uncertainty
from training import train, generate_meta_features, train_ensemble, train_ensemble_no_uncertainty, createSet_baselineModel, generateMeta_features_baseline, createSet_prefix_baseline, generateMeta_features_baseline, generate_meta_features_no_unct
from testing import test_loop, test_loop_ensemble, test_loop_ensemble_no_uncertainty, test_loop_min_uncertainty, test_loop_weighted_average, test_baseline, test_baseline_prefix, test_loop_no_uncertainty, test_loop_average
from models import Ensemble, EnsembleNoUncertainty, SimpleRegressionNoUncertainty, SimpleRegressionUncertainty, Net, NetNoUnc, SimpleEnsemble, SimpleEnsembleNoUncertainty, BranchForEveryInput, StochasticCNN_1D, StochasticCNN_1DNoUnc
import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor
import pickle
from torch import nn, optim
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import KFold
torch.cuda.empty_cache()


'''
    Configurations, hyperparameters and others
'''
path_train = "./RawData/Sepsis Cases_train.xes"
path_val = "./RawData/Sepsis Cases_val.xes"
path_test = "./RawData/Sepsis Cases_test.xes"




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device")
print(device)
learning_rate_lstm = 0.0001
batch_size = 512
slidingWindow = 30


num_epochs_lstm = 30
hidden_size = 128

num_layers = 2


# For ensemble
num_models = 3
num_epochs_ensemble = 451
learning_rate_ensemble = 0.001
learning_rate_ensemble_LP = 0.001
num_epochs_regression = 451
loss_func = nn.MSELoss()
# Load data

num_epochs_cnn = 20
num_filters = 32
filter_size = 3
stride = 2
learning_rate_cnn = 0.001

number_of_runs = 1


train_dataset, val_dataset, test_dataset, num_features_df, features_to_index = load_data(path_train, path_val, path_test, slidingWindow)



printing = False
learning_rate = [0.0001]

for h in range(len(learning_rate)):
    averageMAE_no_no_array = []
    averageMAE_trained_no_array = []
    mae_weighted_average_array = []
    mae_ensemble_no_no_reg_array = []
    mae_ensemble_no_uncertainty_reg_array = []
    mae_ensemble_both_reg_array = []
    mae_ensemble_no_no_mlp_array = []
    mae_ensemble_no_uncertainty_mlp_array = []
    mae_ensemble_both_mlp_array = []
    mae_ensemble_no_no_mlp_simple_array = []
    mae_ensemble_no_uncertainty_mlp_simple_array = []
    mae_ensemble_both_mlp_simple_array = []


    cnn_average_no = []
    cnn_average_with = []


    lstm_average_no = []
    lstm_average_with = []

    baseline_activity_array = []

    print("run number", h)
    with open("results_cnn_lstm_baseline.txt", "a") as file:
        file.write(f"Run number {h} with {learning_rate[h]}" + '\n')

    print("Learning rate")
    print(h)

    learning_rate_ensemble = learning_rate[h]
    learning_rate_ensemble_LP = learning_rate[h]

    learning_rate_branch_gate = learning_rate[h]

    

    embedding_dim = round(math.sqrt(num_features_df))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    '''
        Load Models here. On the first itteration, we train the models. Afterwards, we only do inferencing to save computing ressources.
    '''


    print("LSTM")
    start_train_time = time.time()
    lstm_model = Net(num_features_df, hidden_size, num_layers, embedding_dim).to(device)
    lstm_model =  train(num_epochs_lstm, lstm_model, train_loader, val_loader, learning_rate_lstm)
    end_train_time = time.time()
    training_duration = end_train_time - start_train_time
    print(f"Training completed in {training_duration:.2f} seconds")

    start_inference_time = time.time()
    mae_lstm = test_loop(test_loader, lstm_model, printing=printing)

    end_inference_time = time.time()

    inference_duration = end_inference_time - start_inference_time
    print(f"Inferencing completed in {inference_duration:.4f} seconds")


    # Train LSTM
    #lstm_model_no_unc = NetNoUnc(num_features_df, hidden_size, num_layers, embedding_dim).to(device)
    #lstm_model_no_unc =  train_no_uncertainty(num_epochs_lstm, lstm_model_no_unc, train_loader, val_loader, learning_rate_lstm)
    #mae_lstm_no_uncertainty = test_loop_no_uncertainty(test_loader, lstm_model_no_unc)






