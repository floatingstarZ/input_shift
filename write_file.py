import csv

filename = './catvsdog/submission_example.csv'
with open(filename) as f:
    reader = csv.reader(f)
    csv.writer()
    print(list(reader))