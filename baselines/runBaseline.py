import os

import numpy as np
import torch
import torch.nn.functional as F

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from sklearn.model_selection import learning_curve


# from dataLoader import load, loadCovidTracking

CWD = os.getcwd()
topic = "topic_2"
DATA_PATH_TWITTER = os.path.join(CWD, "TweetSentiment/{}.csv".format(topic))

class LSTM(torch.nn.Module):
    def __init__(self, n_feature, n_hidden=10, n_output=1):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x, _ = self.lstm(x)
        # print(type(x))
        x = self.predict(x)

        return x


def run_linear(X_train, X_test, y_train):
    clf = LinearRegression()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    return y_pred


def run_lr(X_train, X_test, y_train):
    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    
    return y_pred

def run_naive_bayes(X_train, X_test, y_train):
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return y_pred


def main():
    # load the data
    # X, y = loadCovidTracking()

    df = pd.read_csv(DATA_PATH_TWITTER)

    target = "testPositive"
    y = df[[target]].to_numpy()

    sentiment_features = ["AvgSentiment", "Negative", "Positive", "Neutral"]

    use_sentiment_features = False
    related_features = ["death", "totalTested", "hospitalizedIncrease", "temp", "airTravel", "zoom"]

    for rf in related_features:
        if use_sentiment_features:
            features = sentiment_features.copy()
        else:
            features = []

        features.append(rf)
        X = df[features].to_numpy()
        trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

        predY = run_lr(trainX, testX, trainY)

        rmse = np.sqrt(mean_squared_error(testY, predY))

        plt.figure()
        plt.plot(testY, label="gt")
        plt.plot(predY, label="pred")
        plt.legend()
        if use_sentiment_features:
            plt.suptitle('With sentiment - rmse: {}, feature: {}, algo: linear'.format(rmse, rf))
            plt.savefig("baselines/base_models/logistic_regression/{}_{}_sentiment.png".format(rf, '_'.join(sentiment_features)))
        else:
            plt.suptitle('No sentiment (error: {}, feature: {}, algo: linear)'.format(rmse, rf))
            plt.savefig("baselines/base_models/logistic_regression/{}.png".format(rf))

    use_sentiment_features = True
    for rf in related_features:
        if use_sentiment_features:
            features = sentiment_features.copy()
        else:
            features = []

        features.append(rf)
        X = df[features].to_numpy()
        trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

        predY = run_lr(trainX, testX, trainY)

        rmse = np.sqrt(mean_squared_error(testY, predY))

        plt.figure()
        plt.plot(testY, label="gt")
        plt.plot(predY, label="pred")
        plt.legend()
        if use_sentiment_features:
            plt.suptitle('With sentiment - rmse: {}, feature: {}, algo: linear'.format(rmse, rf))
            plt.savefig("baselines/base_models/logistic_regression/{}_{}_sentiment.png".format(rf, '_'.join(sentiment_features)))
        else:
            plt.suptitle('No sentiment (error: {}, feature: {}, algo: linear)'.format(rmse, rf))
            plt.savefig("baselines/base_models/logistic_regression/{}.png".format(rf))

main()


