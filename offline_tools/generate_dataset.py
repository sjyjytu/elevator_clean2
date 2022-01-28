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


class MyQueue:
    def __init__(self):
        self.data = []
        self.idx = 0

    def empty(self):
        return self.idx >= len(self.data)

    def get(self):
        ret = self.data[self.idx]
        self.idx += 1
        return ret

    def put(self, d):
        self.data.append(d)

    def qsize(self):
        return len(self.data) - self.idx


def generate_dataset_from_csv_to_pipline(data_file, dt=0.5, data_of_section=''):
    collected_data = MyQueue()
    used_ele = None
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
        awt, att = 0, 0
        while True:
            time, start_level, end_level, m, standard_ele = row[0], int(row[1].replace('Level ', '')), int(row[2].replace('Level ', '')), float(row[3]),  int(row[7]) - 1
            wt, tt = float(row[10]), float(row[11])
            next_person_time = time.split(':')
            minute, second = next_person_time[-2:]
            nt = int(minute) * 60 + float(second)
            row = next(r)

            if data_of_section != '' and nt < section_start:
                continue
            if data_of_section != '' and nt > section_end:
                break
            collected_data.put({'appear_time': nt, 'start_level': start_level, 'end_level': end_level, 'm': m, 'standard_ele': standard_ele})
            awt += wt
            att += tt
            if last_ele != standard_ele:
                # used_ele.put(standard_ele)
                last_ele = standard_ele

            if len(row) < 14 or row[0]=='':
                break

    awt /= collected_data.qsize()
    att /= collected_data.qsize()
    # print(data_file, awt, att)
    return collected_data, used_ele


def generate_dataset_from_csv_to_txt(data_file, data_of_section='', save_txt=''):
    used_ele = None
    last_ele = -1

    if save_txt == '':
        save_txt = data_file.replace('csv', 'txt')

    if data_of_section != '':
        # '10:00-11:00'
        s, e = data_of_section.split('-')
        sm, ss = s.split(':')
        em, es = e.split(':')
        section_start = int(sm) * 60 + float(ss)
        section_end = int(em) * 60 + float(es)
    with open(data_file, 'rt', encoding='utf8', errors='ignore') as f:
        with open(save_txt, 'w', encoding='utf8', errors='ignore') as sf:
            r = reader(f)
            row = None
            while True:
                if row != None and row != [] and row[0] == 'PASSENGER LIST':
                    break
                row = next(r)
            # #for i in range(274):
            for i in range(10):
                row = next(r)
            awt, att = 0, 0
            while True:
                time, start_level, end_level, m, standard_ele = row[0], int(row[1].replace('Level ', '')), int(row[2].replace('Level ', '')), float(row[3]),  int(row[7]) - 1
                wt, tt = float(row[10]), float(row[11])
                next_person_time = time.split(':')
                minute, second = next_person_time[-2:]
                nt = int(minute) * 60 + float(second)
                row = next(r)

                if data_of_section != '' and nt < section_start:
                    continue
                if data_of_section != '' and nt > section_end:
                    break

                line = ','.join(list(map(str, [nt, start_level, end_level, m, 80, 1.2, 1.2, 1])))
                sf.write(line)
                sf.write('\n')

                if len(row) < 14 or row[0]=='':
                    break


def generate_dataset_from_prob_to_pipline(generate_mask, data_of_section=''):

    if data_of_section != '':
        # '10:00-11:00'
        s, e = data_of_section.split('-')
        sm, ss = s.split(':')
        em, es = e.split(':')
        section_start = int(sm) * 60 + int(ss)
        section_end = int(em) * 60 + int(es)
        t_range = range(section_start, section_end)
    else:
        t_range = range(generate_mask.shape[0])

    collected_data = MyQueue()
    for t in t_range:
        for s in range(generate_mask.shape[1]):
            for d in range(generate_mask.shape[2]):
                if generate_mask[t][s][d]:
                    collected_data.put({'appear_time': t, 'start_level': s+1, 'end_level': d+1, 'm': 75, 'standard_ele': -1})
    return collected_data


def save_dataset(data, file):
    out_put = open(file, 'wb')
    data_str = pickle.dumps(data)
    out_put.write(data_str)
    out_put.close()


def load_dataset(file):
    with open(file, 'rb') as file:
        data = pickle.loads(file.read())
    return data


if __name__ == '__main__':
    # generate_dataset()
    # a = generate_dataset_from_csv_to_pipline('../train_data/up_peak_normal/4Ele16FloorUpPeakFlow28_elvx.csv', data_of_section='20:00-30:00')
    # print(a.get())
    generate_dataset_from_csv_to_txt('../train_data/new/lunchpeak/LunchPeak1_elvx.csv', save_txt='E:\\电梯\\Elevate_Files\\LunchPeak1_elvx.txt')
