import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from torch.autograd import Variable
from config import cfg

src_len = cfg.src_len  # length of source
tgt_len = cfg.tgt_len  # length of target


input_size = cfg.input_size
hidden_size = cfg.hidden_size
n_layers = cfg.n_layers
drop_rate = cfg.drop_rate
batch = cfg.batch
class PhysicalLSTM2(nn.Module):
    def __init__(self, input_dim=cfg.input_size, hidden_size=128, num_layers=1):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)  # Convolutional layer
        self.bn1 = nn.BatchNorm1d(64)  # Batch normalization layer
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)  # Convolutional layer
        self.bn2 = nn.BatchNorm1d(128)  # Batch normalization layer
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)  # Convolutional layer
        self.bn3 = nn.BatchNorm1d(256)  # Batch normalization layer
        self.lstm = nn.LSTM(input_size=256, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(2*hidden_size, 128)  # Fully connected layer
        self.bn4 = nn.BatchNorm1d(128)  # Batch normalization layer
        self.fc2 = nn.Linear(128, 64)  # Fully connected layer
        self.bn5 = nn.BatchNorm1d(64)  # Batch normalization layer
        self.fc3 = nn.Linear(64, 2)  # Fully connected layer
        self.dropout = nn.Dropout(p=0.2)  # Dropout layer
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        yd = x[:, 0, -1]
        time = x[:, 0, 0]
        yd = yd.reshape(yd.shape[0], -1)
        x = x[:, :, :-1]

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        # print(x.shape)

        # Pass through the LSTM layer
        lstm_out, _ = self.lstm(x.transpose(1, 2))  # Transpose x to (batch_size, seq_len, input_size)

        # Flatten LSTM output
        # print(lstm_out.shape)
        # lstm_out = lstm_out.view(lstm_out.size(0), -1)
        lstm_out = lstm_out.reshape(lstm_out.size(0), -1)

        x = self.dropout(self.relu(self.bn4(self.fc1(lstm_out))))
        x = self.dropout(self.relu(self.bn5(self.fc2(x))))
        x = self.sigmoid(self.fc3(x)) * 0.15 + 1.0

        return x, yd, time
