import os, glob, json
import pandas as pd
usefulColumns = ['DATE','LATITUDE','LONGITUDE','ELEVATION','TEMP','TEMP_ATTRIBUTES','DEWP','DEWP_ATTRIBUTES','SLP','SLP_ATTRIBUTES','STP','STP_ATTRIBUTES','VISIB','VISIB_ATTRIBUTES','WDSP','WDSP_ATTRIBUTES','MXSPD','GUST','MAX','MAX_ATTRIBUTES','MIN','MIN_ATTRIBUTES','PRCP','PRCP_ATTRIBUTES','SNDP','FRSHTT','lat','lon','name','admin1','admin2','cc']
locColumns = ['lat','lon','name','admin1','admin2','cc']
def aggStateData(stateDict):
    for state, df in stateDict.items():
        df.drop(locColumns, axis=1, inplace=True)
        df = df.groupby(['DATE']).mean().reset_index()
        df.to_csv("./USData/" + str(state) + ".csv", encoding='utf-8')

def getCountryData(path):
    newFiles = {}
    for fileName in glob.glob(path + "/*.csv"):
        df = pd.read_csv(fileName)
        df = df.loc[df['cc'] == 'US']
        for state in df.admin1.unique():
            currStateData = df.loc[df['admin1'] == state]
            if state in newFiles:
                newFiles[state] = pd.concat([newFiles[state], currStateData[usefulColumns]])
            else:
                newFiles[state] = currStateData[usefulColumns]
    aggStateData(newFiles)



getCountryData("./aggTempData")
