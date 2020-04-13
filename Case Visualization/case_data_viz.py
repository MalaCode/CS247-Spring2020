import csv
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import defaultdict 

def split(date): 
    return [char for char in date] 

def get_dates():
    x = []
    x_int = column_names[0][4:]
    for x_i in x_int:
        date_chars = split(x_i)
        year = 2020
        month = int(date_chars[0])
        day = ""
        day_chars = []
        if date_chars[3] == '/':
            day = int(date_chars[2])
        else:
            day_chars.append(date_chars[2])
            day_chars.append(date_chars[3])
            day = date_chars[2] + date_chars[3]
        day_int = int(day)
        x.append(datetime(year, month, day_int))
    return x
        
def get_country_data(country):
    y_temp= country_data[country]
    y = y_temp[2:]
    y_ints = []
    for i in range(0, len(y)):
        y_i = int(y[i])
        y_ints.append(y_i)
    return y_ints

def plot_data(country):
    plt.plot_date(x, y, linestyle='solid')
    title = country + ' COVID-19 Case Growth'
    plt.title(title)
    plt.ylabel('# Confirmed Cases')
    plt.xlabel('Dates')
    plt.show()
    return

def check_case_doubles(country, dates, cases):
    doubles = 0
    for i in range(0, len(cases)-1):
        if (cases[i+1] >= 2*cases[i]):
            doubles += 1
    print('The number of cases in ' + country + ' has doubled ' + str(doubles) + ' times.')

#################################################################

with open('time_series_covid19_confirmed_global.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    column_names = []
    country_data = defaultdict(list)
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
            column_names.append(row)
        else:
            line_count += 1
            if row[0] == '':
                country_data[row[1]] = row[2:]
            country_data[row[0]] = row[2:]

x = get_dates()
y = get_country_data('Sweden')
check_case_doubles('Sweden', x, y)
plot_data('Sweden')

