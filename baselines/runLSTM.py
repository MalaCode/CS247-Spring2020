import os
import copy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error, mean_squared_error



CWD = os.getcwd()
DATA_PATH = os.path.join(CWD, "unifiedUSData/")
DATA_PATH_COVID = os.path.join(CWD, "CovidTracking/US_historic_data.csv")
topic = "topic_4"
DATA_PATH_TWITTER = os.path.join(CWD, "TweetSentiment/{}.csv".format(topic))

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
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

# get the data frame from csv
df = pd.read_csv(DATA_PATH_TWITTER)

# cor = df.corr()
# sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
# plt.show()
#Correlation with output variable
# cor_target = abs(cor["testPositive"])
# relevant_features = cor_target[cor_target>0]
# print(relevant_features)

# exit()

# choose sentiment features to use from the list ["AvgSentiment", "AvgTopicContrib", "Negative", "Positive", "Neutral"]
# e.g. sentiment_features = ["Negative", "Positive", "Neutral"]
# sentiment_features = ["Negative", "Positive", "Neutral"]
sentiment_features = ["AvgSentiment"]

# the target we want to predict
target = "testPositive"

# whether to use sentiment features
use_sentiment_features = True

# features other than the above sentiment features that we want to use in the training.
# name of the feature is the column header name in the csv file
related_features = ["death", "totalTested", "hospitalizedIncrease", "temp", "airTravel", "zoom"]
# related_features = ["zoom"]
for rf in related_features:
    if use_sentiment_features:
        features = sentiment_features.copy()
    else:
        features = []

    features.append(rf)
    # scale the data
    scaler_t = MinMaxScaler()
    scaler_f = MinMaxScaler()
    target_set = scaler_t.fit_transform(df[[target]])
    features_set = [scaler_f.fit_transform(df[[feature]]) for feature in features]
    features_set = np.transpose(features_set, (1, 0, 2))

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
    num_classes = 1

    lstm = LSTM(num_classes, input_size, hidden_size, num_layers)

    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

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

    # convert torch output to numpy
    data_predict = train_predict.data.numpy()
    dataY_plot = testY.data.numpy()

    # revert the scaling done before the training to display real numbers
    data_predict = scaler_t.inverse_transform(data_predict)
    dataY_plot = scaler_t.inverse_transform(dataY_plot)

    # calculate the root mean squared error for prediction
    rmse = np.sqrt(mean_squared_error(dataY_plot, data_predict))

    plt.figure()
    plt.plot(dataY_plot, label="gt")
    plt.plot(data_predict, label="pred")
    plt.legend()
    if use_sentiment_features:
        plt.suptitle('With sentiment - rmse: {}, feature: {}'.format(rmse, rf))
        plt.savefig("baselines/{}/{}_{}_sentiment.png".format(topic, rf, '_'.join(sentiment_features)))
    else:
        plt.suptitle('No sentiment (error: {}, feature: {})'.format(rmse, rf))
        plt.savefig("baselines/{}/{}.png".format(topic, rf))
    


