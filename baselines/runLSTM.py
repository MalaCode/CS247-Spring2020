import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error, mean_squared_error



CWD = os.getcwd()
DATA_PATH = os.path.join(CWD, "unifiedUSData/")
DATA_PATH_COVID = os.path.join(CWD, "CovidTracking/US_historic_data.csv")
DATA_PATH_TWITTER = os.path.join(CWD, "TweetSentiment/topic_4.csv")

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)#, dropout=0.4)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)
        
        c_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out

def plot_data(data):
    # target_set = df[[target]].values


    plt.plot(data)
    plt.show()


# use twitter sentiment dataset or Covid tracking dataset
use_twitter = True

if not use_twitter:
    df = pd.read_csv(DATA_PATH_COVID)
    df = df.reindex(index=df.index[::-1])
    # replace NaN with 0
    df.fillna(0, inplace=True)
    df.drop(["hash", "dateChecked", "states", "total", "totalTestResults"], axis=1, inplace=True)

    df = df.drop(df.index[:37])

    target = "positive"
    features = ["negative", "hospitalizedCumulative", "inIcuCumulative", "recovered", "death"]
    features = ["positive"]
else:
    df = pd.read_csv(DATA_PATH_TWITTER)

    features = ["AvgSentiment"]
    # features = ["AvgSentiment", "AvgTopicContrib", "Negative", "Positive", "Neutral", "testPositive"]
    # features = ["Negative", "Positive", "Neutral"]
    target = "testPositive"

use_sentiment_features = False
related_features = ["", "death", "totalTested", "hospitalizedIncrease"]
for rf in related_features:
    if rf and use_sentiment_features:
        features.append(rf)
    elif rf and not use_sentiment_features:
        features = [rf]


    # scale the data
    scaler_t = MinMaxScaler()
    scaler_f = MinMaxScaler()
    target_set = scaler_t.fit_transform(df[[target]])
    features_set = [scaler_f.fit_transform(df[[feature]]) for feature in features]
    features_set = np.transpose(features_set, (1, 0, 2))

    # convert to Torch Tensor

    n = target_set.shape[0]
    train_size = int(n * 2 / 3)

    randomize_time_series_data = True
    if randomize_time_series_data:
        # Randomize the data
        trainX, testX, trainY, testY = train_test_split(features_set, target_set, test_size=0.3, random_state=42, shuffle=True)
        trainX = torch.Tensor(trainX)
        testX = torch.Tensor(testX)
        trainY = torch.Tensor(trainY)
        testY = torch.Tensor(testY)
    else:
        # Keep the original order
        trainX, testX = torch.Tensor(features_set[:train_size]), torch.Tensor(features_set[train_size:])
        trainY, testY = torch.Tensor(target_set[:train_size]), torch.Tensor(target_set[train_size:])

    # training
    num_epochs = 5000
    learning_rate = 0.001

    input_size = 1
    hidden_size = 2
    num_layers = 1
    dropout = 0

    num_classes = 1

    lstm = LSTM(num_classes, input_size, hidden_size, num_layers)

    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    #optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        outputs = lstm(trainX)
        optimizer.zero_grad()
        
        loss = criterion(outputs, trainY)
        
        loss.backward()
        
        optimizer.step()
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))


    # Testing
    print("start testing")
    lstm.eval()
    train_predict = lstm(testX)

    data_predict = train_predict.data.numpy()
    dataY_plot = testY.data.numpy()

    data_predict = scaler_t.inverse_transform(data_predict)
    dataY_plot = scaler_t.inverse_transform(dataY_plot)

    # plt.axvline(x=train_size, c='r', linestyle='--')
    print("y", dataY_plot)
    print("pred", data_predict)
    error = mean_squared_error(dataY_plot, data_predict)

    plt.figure()
    plt.plot(dataY_plot, label="gt")
    plt.plot(data_predict, label="pred")
    plt.legend()
    if use_sentiment_features:
        plt.suptitle('LSTM - Tested Positive - Using sentiment (error: {}, feature: {})'.format(error, rf))
        plt.savefig("baselines/lstm_{}_sentiment.png".format(rf))
    else:
        plt.suptitle('LSTM - Tested Positive - No sentiment (error: {}, feature: {})'.format(error, rf))
        plt.savefig("baselines/lstm_{}.png".format(rf))
    


