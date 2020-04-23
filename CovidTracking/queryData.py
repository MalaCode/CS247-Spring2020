import csv
import requests

def getStateHis():
    file_name = "State_historic_data.csv"
    response = requests.get("https://covidtracking.com/api/v1/states/daily.csv")
    decoded_content = response.content.decode('utf-8')
    with open(file_name, "w") as csv_file:
        csv_file.writelines(decoded_content)

def getUSHis():
    file_name = "US_historic_data.csv"
    response = requests.get("https://covidtracking.com/api/v1/us/daily.csv")
    decoded_content = response.content.decode('utf-8')
    with open(file_name, "w") as csv_file:
        csv_file.writelines(decoded_content)


getStateHis()