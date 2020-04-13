import os, glob, json
import pandas as pd


def splitFile(filename, chuncksize):
    fileNum = 0
    for chunk in pd.read_csv(filename, chunksize=chuncksize):
            print('Working on Chunk: ', fileNum)
            chunk.to_csv('./aggTempData/temperatureData' + str(fileNum) + '.csv', encoding='utf-8')
            fileNum += 1

chunksize = 50000
splitFile('./temperatureData.csv', chunksize)