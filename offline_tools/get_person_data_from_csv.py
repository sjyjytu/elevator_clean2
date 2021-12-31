import os
import numpy as np
import pickle
import glob
from csv import reader


def get_person_data():
    data_f = 'person_flow_data'
    all_files = glob.glob(os.path.join(data_f, '*'))
    for file in all_files:
        f = open(file, 'rt', encoding='utf8', errors='ignore')
        collected_data = []
        r = reader(f)
        row = None
        for i in range(556):
            row = next(r)
        while True:
            time, start_level, end_level = row[0], row[1].replace('Level ', ''), row[2].replace('Level ', '')
            row = next(r)
            if len(row) != 15:
                break
            collected_data.append([time, start_level, end_level])
        out_file_name = file.replace('csv', 'pkl')
        pickle.dump(collected_data, open(out_file_name, 'wb'))
    print(all_files)


if __name__ == '__main__':
    get_person_data()
