import os, glob, json
import pandas as pd
import ntpath
import numpy as np
from datetime import datetime

stateABVs = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}

def addSingularValues(stateDict, inputDF, stateColName, newColName):
    for state, df in stateDict.items():
        colsToAdd = inputDF.loc[inputDF[stateColName] == state]
        colsToAdd = colsToAdd.drop([stateColName],axis=1)
        for col in colsToAdd.columns:
            if(len(colsToAdd[col].values) > 0):
                stateDict[state][col] = colsToAdd[col].values[0]
            else:
                print("Minor error processing state " + state + ", found columns with 0 length, adding NaN column...")
                stateDict[state][col] = np.nan
    return

def addDateValues(stateDict, inputDF, stateColName, joinColNames):
    for state, df in stateDict.items():
        colsToAdd = inputDF.loc[inputDF[stateColName] == state]
        # colsToAdd = colsToAdd.drop([stateColName],axis=1)
        stateDict[state] = pd.merge(stateDict[state], colsToAdd, how='left', on=joinColNames)
        stateDict[state] = stateDict[state].drop([stateColName], axis=1)
    return

def getAirTravelData(stateDict):
    airFile = "./AirTravel/flightData.csv"
    df = pd.read_csv(airFile)
    df = df.rename(columns={'Unnamed: 0': 'State'})
    addSingularValues(stateDict, df, 'State', df.columns)

def getHospitalData(stateDict):
    hospFile = "./Hospital Numbers/HospitalsPerState.csv"
    df = pd.read_csv(hospFile)
    addSingularValues(stateDict, df, 'State', df.columns)

def getSAHDate(stateDict):
    sahFile = "./Policy Data/stay_at_home_dates.csv"
    df = pd.read_csv(sahFile)
    addSingularValues(stateDict, df, 'State', df.columns)

def getPopDensity(stateDict):
    popDenFile = "./Population Density/PopulationDensity.csv"
    df = pd.read_csv(popDenFile)
    addSingularValues(stateDict, df, 'State', df.columns)

def getPoliticSide(stateDict):
    polSideFile = "./StatePoliticSide/data.csv"
    df = pd.read_csv(polSideFile, usecols=["State","pvi","governorParty","senateParty"])
    addSingularValues(stateDict, df, 'State', df.columns)

def getUSRegion(stateDict):
    regionFile = "./USRegions/USRegions.csv"
    df = pd.read_csv(regionFile)
    addSingularValues(stateDict, df, 'State', df.columns)

def getCovidTracking(stateDict):
    covidFile = "./CovidTracking/State_historic_data.csv"
    df = pd.read_csv(covidFile)
    df['date'] = df['date'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d').strftime('%Y-%m-%d'))
    df['state'] = df['state'].apply(lambda x: stateABVs[x])
    df = df.rename(columns={'date':'DATE'})
    addDateValues(stateDict,df,'state',['DATE'])
    # for state, sdf in stateDict.items():
    #     stateDict[state] = sdf[df.columns]

def getTemperatureData(path):
    stateDict = {}
    for fileName in glob.glob(path + "/*.csv"):
        state = os.path.splitext(ntpath.basename(fileName))[0]
        stateDict[state] = pd.read_csv(fileName)
    return stateDict



def aggAllData():
    stateDict = getTemperatureData('./Temperature/USData')
    print("Adding Flight Data")
    getAirTravelData(stateDict)
    print("Adding Hospital Data")
    getHospitalData(stateDict)
    print("Adding SAH Dates")
    getSAHDate(stateDict)
    print("Adding Population Density")
    getPopDensity(stateDict)
    print("Adding State Political Side")
    getPoliticSide(stateDict)
    print("Adding US Regions")
    getUSRegion(stateDict)
    print("Adding Covid Data")
    getCovidTracking(stateDict)
    allStates = pd.DataFrame()
    for state, df in stateDict.items():
        df.to_csv("./unifiedUSData/" + str(state) + ".csv", encoding='utf-8')
        allStates = pd.concat([allStates, df])
    df.to_csv("./unifiedUSData/allStates.csv", encoding='utf-8')

aggAllData()
