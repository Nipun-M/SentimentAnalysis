import csv
import pandas as pd

def get_dataframe(filename):
    raw_data = dict()
    reader = csv.DictReader(open(filename))
    headers = list(reader.__next__().keys())
    for header in headers:
        raw_data[header] = list()
    for row in reader:
        for header in headers:
            raw_data[header].append(row[header])
    df = pd.DataFrame(raw_data)
    return df