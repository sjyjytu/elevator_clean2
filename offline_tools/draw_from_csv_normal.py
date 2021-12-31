import os
import numpy as np
import pickle
import glob
from csv import reader
import pandas as pd

import matplotlib.pyplot as plt


def get_person_data():
    data_f = 'E:\\电梯\\0923\\给交大数据\\DownPeakFlow\\Normal'
    path = os.path.join(data_f, '*')
    all_files = glob.glob(path)
    for file in all_files:
        f = open(file, 'rt', encoding='utf8', errors='ignore')
        collected_data = []
        r = reader(f)
        row = None
        for i in range(274):
            row = next(r)
        while True:
            # print(row)
            time, start_level, end_level = row[0], row[1].replace('Level ', ''), row[2].replace('Level ', '')
            s_time = time.split(':')
            time_idx = float(s_time[1]) * 60 + float(s_time[2])
            print(time_idx)
            row = next(r)
            if len(row) != 15:
                break
            collected_data.append([time_idx, start_level, end_level])
        # out_file_name = file.replace('csv', 'pkl')
        # pickle.dump(collected_data, open(out_file_name, 'wb'))
        df = pd.DataFrame(collected_data)
        # print(len(collected_data))
        df.rename(columns={0:'tidx',1:'src', 2:'dst'},inplace=True)
        g = df.groupby(['tidx', 'src'], as_index=False).count()
        # df.groupby(['tidx', 'src'])['tidx', 'src'].agg({'cnt': 'count'})
        # print(df)
        # print(g)
        tcount = len(df.groupby(['tidx']))
        yss = [[0 for t in range(tcount)] for _ in range(17)]
        tno = -1
        ltidx = -10.
        for i, r in g.iterrows():
            tidx, src, dst = int(r['tidx']), int(r['src']), int(r['dst'])
            if ltidx != tidx:
                ltidx = tidx
                tno += 1
                print(tno, tidx//60, ':', tidx%60)
                if tno > 0:
                    for ri in range(17):
                        yss[ri][tno] = yss[ri][tno-1]  # accumulate
            yss[src-1][tno] += dst

        ts = list(range(tcount))
        for pi in range(0, 17):
            # if pi % 2 == 1:
            plt.scatter(ts, yss[pi], alpha=0.6, label='level %d'%(pi+1))  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
            # plt.plot(ts, yss[pi], alpha=0.6, label='level %d'%(pi+1))  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
        plt.legend()
        plt.show()
        break
    # print(all_files)


if __name__ == '__main__':
    get_person_data()
