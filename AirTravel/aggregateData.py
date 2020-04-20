import os, glob, json
import pandas as pd

def aggData(filename):
    res = pd.DataFrame()
    currFile = pd.read_csv(filename)
    print(currFile)

    res["NumInFlights"] = currFile['DEST_STATE_NM'].value_counts()
    res["NumOutFlights"] = currFile['ORIGIN_STATE_NM'].value_counts()

    res["NumDomesticInFlights"] = currFile.loc[currFile['ORIGIN_COUNTRY'] == 'US', 'DEST_STATE_NM'].value_counts()
    res["NumInternationalInFlights"] = currFile.loc[currFile['ORIGIN_COUNTRY'] != 'US', 'DEST_STATE_NM'].value_counts()

    res["NumDomesticOutFlights"] = currFile.loc[currFile['DEST_COUNTRY'] == 'US', 'ORIGIN_STATE_NM'].value_counts()
    res["NumInternationalOutFlights"] = currFile.loc[currFile['DEST_COUNTRY'] != 'US', 'ORIGIN_STATE_NM'].value_counts()

    res["TotalPassengers"] = currFile.groupby('DEST_STATE_NM').PASSENGERS.sum()
    # res["TotalInPassengers"] = currFile.loc[currFile['DEST_COUNTRY'] == 'US', ['ORIGIN_STATE_NM', 'PASSENGERS']].groupby('ORIGIN_STATE_NM').sum()
    # res["TotalOutPassengers"] = currFile.loc[currFile['DEST_COUNTRY'] != 'US', ['ORIGIN_STATE_NM', 'PASSENGERS']].groupby('ORIGIN_STATE_NM').sum()
    res.fillna(0, inplace=True)
    res.sort_index(inplace=True)

    res["TotalFlights"] = res['NumInFlights'] + res['NumOutFlights']
    res.to_csv('flightData.csv', encoding='utf-8')



aggData("./704304538_T_T100_SEGMENT_ALL_CARRIER.csv")