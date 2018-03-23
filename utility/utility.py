import csv
import pandas as pd


def get_dataframe(filename):
    raw_data = dict()
    reader = csv.reader(open(filename))
    headers = list(reader.__next__())
    for header in headers:
        raw_data[header] = list()
    for row in reader:
        raw_data['sentiment'].append(row[0])
        raw_data['content'].append(','.join(row[1:]))
    df = pd.DataFrame(raw_data)
    return df