import os
import sys
import glob
import pickle
import random
from smec_liftsim.utils import EPSILON
from smec_liftsim.utils import PersonType
from smec_liftsim.mansion_configs import MansionConfig
from smec_liftsim.person_generators import PersonGeneratorBase

from offline_tools.generate_dataset import *
from draw_data import process2, dt as delta_t
import numpy as np


class RandomDataGenerator(PersonGeneratorBase):
    """
    Fetch data from disk and generate person data.
    """
    def __init__(self, data_dir=None, data_of_section='', random_or_load_or_save=0):
        super().__init__()
        # print(data_file, data_dir, file_begin_idx)
        self._cur_id = 0
        self.reset_cnt = 0
        self.data_of_section = data_of_section
        self.data_dir = data_dir
        self.random_or_load_or_save = random_or_load_or_save
        assert self.data_dir is not None

        dataX = process2(data_dir=self.data_dir)  # file, t, s, d
        dataX = np.average(dataX, axis=0)  # 均值 t,s,d  每分钟的人数
        self.prob = dataX / delta_t  # 每分钟中，每秒的生成人的概率  t,s,d

        # generate dataset
        self.T, self.S, self.D = dataX.shape
        self.generate_mask = np.random.random((delta_t, self.T, self.S, self.D))
        self.generate_mask = self.generate_mask < self.prob  # delta_t, t, s, d
        self.generate_mask = self.generate_mask.transpose((1,0,2,3))
        self.generate_mask = self.generate_mask.reshape((-1, self.S, self.D))

        self.data_of_section = data_of_section
        self.data = generate_dataset_from_prob_to_pipline(self.generate_mask, self.data_of_section)
        if self.random_or_load_or_save == 1:
            self.data = load_dataset(f'./save_datasets/{self.reset_cnt}')
        elif self.random_or_load_or_save == 2:
            save_dataset(self.data, f'./save_datasets/{self.reset_cnt}')
        self.total_person_num = self.data.qsize()
        self.next_person = None if self.data.empty() else self.data.get()
        self.done = False

    def reset(self):
        # generate dataset
        self._cur_id = 0
        self.reset_cnt += 1
        self.generate_mask = np.random.random((delta_t, self.T, self.S, self.D))
        self.generate_mask = self.generate_mask < self.prob  # delta_t, t, s, d
        self.generate_mask = self.generate_mask.transpose((1, 0, 2, 3))
        self.generate_mask = self.generate_mask.reshape((-1, self.S, self.D))

        self.data = generate_dataset_from_prob_to_pipline(self.generate_mask, self.data_of_section)
        if self.random_or_load_or_save == 1:
            self.data = load_dataset(f'./save_datasets/{self.reset_cnt}')
        elif self.random_or_load_or_save == 2:
            save_dataset(self.data, f'./save_datasets/{self.reset_cnt}')

        self.total_person_num = self.data.qsize()
        self.next_person = None if self.data.empty() else self.data.get()
        self.done = False

    def generate_person(self):
        """
        Generate Random Persons from Poisson Distribution
        Returns:
          List of Random Persons
        """
        ret_persons = []
        cur_time = self._config.raw_time

        while self.next_person is not None and cur_time >= self.next_person['appear_time']:
            # generate this person
            ret_persons.append(
                PersonType(
                    self._cur_id,
                    self.next_person['m'],
                    self.next_person['start_level'],
                    self.next_person['end_level'],
                    cur_time,
                    self.next_person['standard_ele']
                ))
            self._cur_id += 1
            if not self.data.empty():
                self.next_person = self.data.get()
            else:
                self.next_person = None
                self.done = True
        return ret_persons


if __name__ == '__main__':
    gen = RandomDataGenerator(data_dir='../train_data/new/lunchpeak', data_of_section='00:00-06:00')
    print(gen.data.qsize())
    for d in gen.data.data:
        print(d)
    # print(gen.data.data)
    # a = gen.generate_person()
