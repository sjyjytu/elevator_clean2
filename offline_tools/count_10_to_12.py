import os
import numpy as np
import pickle
import glob
from csv import reader
import pandas as pd

import matplotlib.pyplot as plt


def get_person_data():
    data_f = '../person_flow_data'
    path = os.path.join(data_f, '*')
    all_files = glob.glob(path)
    cnt = [[0 for li in range(17)] for _ in range(17)]
    for file in all_files:
        if not file.endswith('csv'):
            continue
        # print(file)
        f = open(file, 'rt', encoding='utf8', errors='ignore')
        r = reader(f)
        row = None
        for i in range(556):
            row = next(r)

        while True:
            time, start_level, end_level = row[0], row[1].replace('Level ', ''), row[2].replace('Level ', '')
            row = next(r)
            s_time = time.split(':')
            h = int(s_time[0])
            # m = int(s_time[1])
            if h < 10:
                continue
            if h > 12:
                break
            cnt[int(start_level)-1][int(end_level)-1] += 1
            # time_idx = int(s_time[0]) * 60 + int(s_time[1]) // 5
            # print(time_idx)

            if len(row) != 15:
                break
        # break
    for i in range((17)):
        for j in range((17)):
            cnt[i][j] /= (30 * 3 * 60)
    print(cnt)


if __name__ == '__main__':
    get_person_data()
