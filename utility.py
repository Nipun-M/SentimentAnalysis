import csv

def load_data_set(filename):
    csv_reader = csv.reader(open(filename))
    data = list()
    for row in csv_reader:
        senti = row[0]
        content = ','.join(row[1:])
        data.append((senti, content))
    data.pop(0)
    return data