# Uncertainty aware ensemble methods for predictive process monitoring
Scripts for the Masterthesis: "Uncertainty aware ensembles methods for predictive process monitoring"


## General:
Master Thesis - Process Mining Codebase.
This repository contains code and datasets used for process mining experiments conducted for my master's thesis. It covers event log preprocessing, process model discovery, and evaluation methods.


## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Output Files](#output)
4. [Reading Results](#reading)



## Installation
1. Clone the repository:
   ```bash
    git clone https://github.com/HenrikMader/Masterthesis.git

2. Create virtual environment & activate it
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install requirements.txt
    ```bash
    pip install -r requirements.txt

## Usage
1. Create test and train Split.

    - Go to the Folder DataPreparation
    - Put the xes log file into the Raw Data for which you want to do the prediction (You will already find Sepsis Cases there right now)
    - Run the script (e.g sepsis.py)
    - A train and a test file will be created in RawData. Note, that you need to manually create the validation file! We always took the last 15% of the traces from the train dataset.
2. Run the main script
    - Navigate to main.py
    - Change the train, test and validation paths. 
    - Important: When you want to use models which are already trained (e.g from Google Drive), then you need to select them accordingly:

    Do this with:

    ```bash
    cnn_model = torch.load("./path_to_your_model")
    ```

    Instead of this:
    cnn_model = train(num_epochs_cnn, cnn_model, train_loader, val_loader, learning_rate_cnn)

    Tune the Hyperparameters accordingly. 
    Important: With numberOfRuns you can say how often you want the ensembles to produce results across the learning rates. Select 1 if this should only run once
    Select printing = True if you want to create plots (e.g the plots of the Accuracy Rejection Curves)

## Output files
Once the main.py is finished, there are different files created.

1. train.pt, val.pt and test.pt are the files from the encoding. Encoding with large datasets takes a while (e.g on 2012). You can load them under dataPreparation.py with e.g

instead of: train_dataset = creatingTensorsTrainingAndTesting(train_df, feature_to_index, sliding_window=slidingWindow)
say: train:dataset = torch.load("./train.pt)

2. The models from the ensemble (e.g EnsembleNoUnc). Those files are less important, because the ensembles are training fast

3. results_lstm_cnn_baseline.txt: In this file you will find all of the MAE results across the different Models and learning rates.


## Reading results

You will see on every learning rate different results, from the base Models and from the Ensembles (Average, Regression, Simple MLP and Complex MLP). MLP means neural network for the results. The first Word always suggest which model was running. E.g:

- MLP Simple not trained and not given: 7.65

means that the Simple MLP which was not trained on the heteroscedasticity and bayes assumption and the uncertainty was not given  ensemble (no, no) was 7.65

- MLP Simple trained and not given: xxx

means that the MLP was trained with heteroscedasticity and bayes but the uncertainty Information was not given ensemble (yes, no)

- MLP Simple trained and given uncertainty 

is the ensemble (yes, yes) Group.