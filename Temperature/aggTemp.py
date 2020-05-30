import glob 
import pandas as pd

def aggTemp(path):
    df = pd.DataFrame()
    for f in glob.glob(path):
        tdf = pd.read_csv(f,usecols=['DATE', 'TEMP'])
        tdf['DATE'] = pd.to_datetime(tdf['DATE'])
        tdf = tdf[tdf['DATE'].dt.month == 3]
        df = df.append(tdf)
    final = df.groupby(by='DATE').mean()
    final.to_csv('./AvgTempPerDay.csv')
aggTemp('./USData/*.csv')