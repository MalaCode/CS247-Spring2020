import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from sklearn.model_selection import learning_curve


from dataLoader import load, loadCovidTracking

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden=10, n_output=1):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

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

def run_nn(X_train, X_test, y_train, y_test):
    # print(X_train.shape)
    # print(X_train)
    n_features = X_train.shape[1]
    # net = Net(n_feature=n_features)
    net = LSTM(n_feature=n_features)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    loss_func = torch.nn.MSELoss() 

    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)

    epochs = 100
    
    for e in range(epochs):
        running_loss = 0
        for i in range(len(X_train)):
            test = X_train[i].view(1, 1, n_features)
            # print(type(test), test.shape, test)
            
            prediction = net(test)     # input x and predict based on x
            # print(prediction)
            # print(y_train[i])

            loss = loss_func(prediction, y_train[i].view(1, 1, 1))     # must be (1. nn output, 2. target)

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            running_loss += loss.item()
            optimizer.step()        # apply gradients
        print(e, running_loss/len(X_train))
    # print(X_test[0], X_train[0])
    # print(net(X_test[0]))
    print(X_test)
    y_pred = [net(X_test[i].view(1, 1, n_features)) for i in range(len(X_test))]
    print(y_pred)
    # print(running_loss)

    

def run_linear(X_train, X_test, y_train, y_test):
    clf = LinearRegression()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    plt.plot(y_test, label="gt")
    plt.plot(y_pred, label="pred")
    plt.legend()
    plt.suptitle('Linear Regression - Tested Positve')
    plt.savefig("baselines/linear.png")
    # print(y_test, y_pred)
    log_error = mean_squared_log_error(y_test, y_pred)
    error = mean_squared_error(y_test, y_pred)
    print("Linear Regression: MSE:", error, "; log:", log_error)


def run_lr(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    # print(y_pred)
    error = mean_squared_log_error(y_test, y_pred)
    print("Logistic Regression", error)

def run_naive_bayes(X_train, X_test, y_train, y_test):
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    error = mean_squared_log_error(y_test, y_pred)
    # print(y_test, y_pred)
    print("Naive Bayes: ", error)


def main():
    # load the data
    X, y = loadCovidTracking()
    # print(len(X))
    # print(y)
    # exit()
    n = y.shape[0]
    train_size = int(n * 2 / 3)

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    print("size", train_size)

    # split data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)
    # print(X_train.shape, X_train)
    # print(y_train.shape, y_train)
    # print(X_train.shape, X_train)
    # print(X_train.shape, X_train)

    run_linear(X_train, X_test, y_train, y_test)
    # run_lr(X_train, X_test, y_train, y_test)
    # run_naive_bayes(X_train, X_test, y_train, y_test)
    # run_nn(X_train, X_test, y_train, y_test)

# test = torch.cat([torch.randn(1, 3) for _ in range(5)])
# print(test.shape, test)
# print(test.view(len(test), 1, -1), test.view(len(test), 1, -1).shape)
main()


