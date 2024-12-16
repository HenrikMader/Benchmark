
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.model_selection import KFold, train_test_split
from math import sqrt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import time

dataset = pd.read_csv('./CleanedData/cleanData_Final.csv')
X = dataset[['PrevAVGCost', 'PrevAssignedCost', 'AVGCost', 'LatestDateCost', 'A', 'B', 'C', 'D', 'E', 'F', 'G']].values
y = dataset['GenPrice'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)


# Prepare input (X) and target (y)
X = dataset[['PrevAVGCost', 'PrevAssignedCost', 'AVGCost', 'LatestDateCost', 'A', 'B', 'C', 'D', 'E', 'F', 'G']].values
y = dataset['GenPrice'].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Make y a column vector

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(11, 64)  # 11 input features, 64 hidden units
        self.fc2 = nn.Linear(64, 128)  # Hidden layer
        self.fc3 = nn.Linear(128, 64)   # Output layer
        self.fc2 = nn.Linear(64, 32)  # Hidden layer
        self.fc3 = nn.Linear(32, 1)   # Output layer


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model, loss function, and optimizer
model = NeuralNetwork()
criterion = nn.L1Loss()  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

start_train_time = time.time()

# Training the model
num_epochs = 500
for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")


end_train_time = time.time()
training_duration = end_train_time - start_train_time
print(f"Training completed in {training_duration:.2f} seconds")



# Evaluate the model
model.eval()

start_inference_time = time.time()

with torch.no_grad():
    y_test_pred = model(X_test)
    test_loss = criterion(y_test_pred, y_test)
    print(f"Test Loss: {test_loss.item():.4f}")

end_inference_time = time.time()
inference_duration = end_inference_time - start_inference_time
print(f"Inferencing completed in {inference_duration:.4f} seconds")

# Convert predictions back to numpy for further analysis
y_test_pred = y_test_pred.numpy()
y_test = y_test.numpy()






