from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import mean_squared_log_error


from dataLoader import load


def run_lr(X_train, X_test, y_train, y_test):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(y_pred)
    error = mean_squared_log_error(y_test, y_pred)
    print(error)

def run_naive_bayes(X_train, X_test, y_train, y_test):
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    error = mean_squared_log_error(y_test, y_pred)
    print(y_test, y_pred)
    print(error)


def main():
    # load the data
    X, y = load()

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=False)
    print(X_train.shape, X_train)
    print(y_train.shape, y_train)
    # print(X_train.shape, X_train)
    # print(X_train.shape, X_train)

    run_lr(X_train, X_test, y_train, y_test)

main()


