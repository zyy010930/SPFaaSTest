import csv
import math
import random
import pandas as pd

with open('./SPFaaS_pre.csv', 'r', encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]

path = './newData.csv'
dt = pd.read_csv(path, header=0)
dt = dt.iloc[:, :1440]

hit_0 = 0
sum_0 = 0
hit_1 = 0
sum_1 = 0

for i in range(len(rows)):
    for j in range(1393):
        pred = 0
        for k in range(5):
            if float(rows[i][j*5+k]) >= 1:
                pred = 1
                break
        if dt.iloc[i, j+43] != 0:
            sum_1 += 1
            if pred == 1:
                hit_1 += 1
        else:
            sum_0 += 1
            if pred == 0:
                hit_0 += 1

print(sum_1)
print(sum_0)
print(hit_1/sum_1)
print(hit_0/sum_0)
