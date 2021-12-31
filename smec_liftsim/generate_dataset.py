import os
import numpy as np
import pickle
import glob
from csv import reader
import queue


def generate_dataset():
    data_f = './person_flow_data'
    all_files = glob.glob(os.path.join(data_f, '*'))
    collected_data = []
    day = -1
    for file in all_files:
        if not file.endswith('.csv'):
            continue
        day += 1
        f = open(file, 'rt', encoding='utf8', errors='ignore')
        r = reader(f)
        row = None
        for i in range(556):
            row = next(r)
        while True:
            time, start_level, end_level = row[0], row[1].replace('Level ', ''), row[2].replace('Level ', '')
            next_person_time = time.split(':')
            nt = (int(next_person_time[0]) - 7) * 3600 + int(next_person_time[1]) * 60 + float(next_person_time[2])
            row = next(r)
            if len(row) != 15:
                break
            collected_data.append([day, nt, start_level, end_level])
        out_file_name = file.replace('csv', 'pkl')
        # pickle.dump(collected_data, open(out_file_name, 'wb'))
    # print(collected_data)
    # print(len(collected_data))  # 319513
    return collected_data


def generate_dataset_from_csv_to_pipline(data_file, dt=0.5, data_of_section=''):
    collected_data = queue.Queue()
    used_ele = queue.Queue()
    last_ele = -1
    if data_of_section != '':
        # '10:00-11:00'
        s, e = data_of_section.split('-')
        sm, ss = s.split(':')
        em, es = e.split(':')
        section_start = int(sm) * 60 + float(ss)
        section_end = int(em) * 60 + float(es)
    with open(data_file, 'rt', encoding='utf8', errors='ignore') as f:
        r = reader(f)

        row = None
        while True:
            if row != None and row != [] and row[0] == 'PASSENGER LIST':
                break
            row = next(r)
        # #for i in range(274):
        for i in range(10):
            row = next(r)

        # for i in range(274):
        #     row = next(r)
        while True:
            time, start_level, end_level, m, standard_ele = row[0], int(row[1].replace('Level ', '')), int(row[2].replace('Level ', '')), float(row[3]),  int(row[7]) - 1
            next_person_time = time.split(':')
            minute, second = next_person_time[-2:]
            nt = int(minute) * 60 + float(second)
            collected_data.put({'appear_time': nt, 'start_level': start_level, 'end_level': end_level, 'm': m,
                                'standard_ele': standard_ele})
            used_ele.put(standard_ele)
            row = next(r)

            if len(row) < 14 or row[0] == '':
                break
            if data_of_section != '':
                if nt < section_start:
                    continue
                elif nt > section_end:
                    break

            #if last_ele != standard_ele:

            #    last_ele = standard_ele
    return collected_data, used_ele


if __name__ == '__main__':
    # generate_dataset()
    a, b = generate_dataset_from_csv_to_pipline('const3.csv')
    while not a.empty():
        print(a.get())
