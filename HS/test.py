import numpy as np
import csv

data = {}
with open('DT/example.csv', 'r') as csvFile:
    csvReader = csv.reader(csvFile)
    for num, row in enumerate(csvReader):
        if num > 0:
            data[num] = [row[7], row[9]]

print(data)
