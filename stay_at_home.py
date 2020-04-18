import csv
from datetime import date

with open('stay_at_home.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(["State", "Date Order Placed"])
    writer.writerow(["Alabama", date(2020, 4, 3)])
    writer.writerow(["Alaska", date(2020, 3, 28)])
    writer.writerow(["Arizona", date(2020, 3, 31)])
    writer.writerow(["California", date(2020, 3, 19)])
    writer.writerow(["Colorado", date(2020, 3, 26)])
    writer.writerow(["Connecticut", date(2020, 3, 23)])
    writer.writerow(["Delaware", date(2020, 3, 24)])
    writer.writerow(["Florida", date(2020, 4, 3)])
    writer.writerow(["Georgia", date(2020, 4, 3)])
    writer.writerow(["Hawaii", date(2020, 3, 25)])
    writer.writerow(["Idaho", date(2020, 3, 25)])
    writer.writerow(["Illinois", date(2020, 3, 21)])
    writer.writerow(["Indiana", date(2020, 3, 24)])
    writer.writerow(["Kansas", date(2020, 3, 30)])
    writer.writerow(["Kentucky", date(2020, 3, 26)])
    writer.writerow(["Louisiana", date(2020, 3, 23)])
    writer.writerow(["Maine", date(2020, 4, 2)])
    writer.writerow(["Maryland", date(2020, 3, 30)])
    writer.writerow(["Massachusetts", date(2020, 3, 24)])
    writer.writerow(["Michigan", date(2020, 3, 24)])
    writer.writerow(["Minnesota", date(2020, 3, 27)])
    writer.writerow(["Mississippi", date(2020, 4, 3)])
    writer.writerow(["Missouri", date(2020, 4, 6)])
    writer.writerow(["Montana", date(2020, 3, 28)])
    writer.writerow(["Nevada", date(2020, 4, 1)])
    writer.writerow(["New Hampshire", date(2020, 3, 27)])
    writer.writerow(["New Jersey", date(2020, 3, 21)])
    writer.writerow(["New Mexico", date(2020, 3, 24)])
    writer.writerow(["New York", date(2020, 3, 22)])
    writer.writerow(["North Carolina", date(2020, 3, 30)])
    writer.writerow(["Ohio", date(2020, 3, 23)])
    writer.writerow(["Oklahoma", None])
    writer.writerow(["Oregon", date(2020, 3, 23)])
    writer.writerow(["Pennsylvania", date(2020, 4, 1)])
    writer.writerow(["Rhode Island", date(2020, 3, 28)])
    writer.writerow(["South Carolina", date(2020, 4, 7)])
    writer.writerow(["Tennessee", date(2020, 4, 2)])
    writer.writerow(["Texas", date(2020, 4, 2)])
    writer.writerow(["Virgina", date(2020, 3, 30)])
    writer.writerow(["Vermont", date(2020, 3, 25)])
    writer.writerow(["Washington", date(2020, 3, 23)])
    writer.writerow(["West Virginia", date(2020, 3, 24)])
    writer.writerow(["Wisconsin", date(2020, 3, 25)])