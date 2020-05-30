import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils.validation import column_or_1d


CWD = os.getcwd()
DATA_PATH = os.path.join(CWD, "unifiedUSData/")
DATA_PATH_COVID = os.path.join(CWD, "CovidTracking/US_historic_data.csv")

def load(state="New York", target="positive"):
    df = pd.read_csv(DATA_PATH + state + ".csv")
    # df = pd.read_csv(DATA_PATH)

    # replace NaN with 0
    df.fillna(0, inplace=True)

    # to generate the correlation heatmap
    plt.figure(figsize=(12,10))
    cor = df.corr()
    # sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    # plt.show()
    #Correlation with output variable
    cor_target = abs(cor["positive"])
    #Selecting highly correlated features
    relevant_features = cor_target[cor_target>0.5]
    print(relevant_features)

    # Drop state-dependent columns if we care about the entire US.
    # if state == "allStates":
    col_to_drop = [3, 4, 5] + list(range(25, 44))
    df.drop(df.columns[col_to_drop], axis=1, inplace=True)

    # drop the first two indexing columns + dates column
    df.drop(df.columns[[0, 2]], axis=1, inplace=True)
    
    # Drop "hash" and "dateChecked" columns
    df.drop(["hash", "dateChecked"], axis=1, inplace=True)

    # print(df.head())

    # get the target 
    y = df[[target]].to_numpy()
    y = column_or_1d(y, warn=True)

    # drop the target column for X.
    X = df.drop([target], axis=1, inplace=True)
    num_col = len(df.columns)
    X = df[list(df.columns)].to_numpy()
 
    # For testing purpose only: keep only the relevant one feature
    X = df[df.columns[[0]]].to_numpy()

    return X, y


def loadCovidTracking(target="positive"):
    df = pd.read_csv(DATA_PATH_COVID)
    df = df.reindex(index=df.index[::-1])

    # replace NaN with 0
    df.fillna(0, inplace=True)
    df.drop(["hash", "dateChecked", "states", "total", "totalTestResults"], axis=1, inplace=True)

    df = df.drop(df.index[:37])

    # X = df[df.columns[[0]]].to_numpy()
    y = df[[target]].to_numpy()

    # df.drop([target], axis=1, inplace=True)
    # X = df[list(df.columns)].to_numpy()
    # X = df[["negative", "hospitalizedCumulative", "inIcuCumulative", "recovered", "death"]].to_numpy()
    X = df[["positive"]].to_numpy()

    # correlation
    cor = df.corr()
    # sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    # plt.show()
    #Correlation with output variable
    cor_target = abs(cor["positive"])
    #Selecting highly correlated features
    relevant_features = cor_target[cor_target>0.5]
    print(relevant_features)

    return X, y



loadCovidTracking()