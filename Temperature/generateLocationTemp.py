import os, glob, json
import pandas as pd
import tqdm
import reverse_geocoder as rg

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import GoogleV3

def generateLocationTempFile(locator ,path):
    numFiles = 0
    res = pd.DataFrame()

    for fileName in glob.glob(path + "/*.csv"):
        if(numFiles % 20 == 0):
            print("Working on: ", fileName)
        currFile = pd.read_csv(fileName)
        currFile.drop(['STATION', 'NAME'], axis=1, inplace=True)
        if(currFile['LATITUDE'].dropna().empty or currFile['LONGITUDE'].dropna().empty):
            print("NO LATITUDE OR LONGITUDE FOUND IN FILE: " , fileName)
            continue
        coordinates = (currFile['LATITUDE'][0], currFile['LONGITUDE'][0])
        # print(fileName, coordinates)
        locInfo = rg.search(coordinates)
        if(len(locInfo) < 1):
            numFiles += 1
            print(fileName)
            continue
        locDF = pd.DataFrame(locInfo, columns=locInfo[0].keys())
        locDF['key'] = 1
        currFile['key'] = 1
        currFile = pd.merge(currFile, locDF, on='key').drop('key',axis=1)
        res = res.append(pd.concat([currFile], axis=1))
        numFiles += 1    
    res.to_csv('temperatureData.csv', encoding='utf-8')
    return

locator = Nominatim(user_agent="myGeocoder", timeout=10)
rgeocode = RateLimiter(locator.reverse, min_delay_seconds=1)
generateLocationTempFile(rgeocode, "./tempData")
