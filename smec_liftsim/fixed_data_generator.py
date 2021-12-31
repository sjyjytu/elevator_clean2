import os
import sys
import glob
import pickle
import random
from smec_liftsim.utils import EPSILON
from smec_liftsim.utils import PersonType
from smec_liftsim.mansion_configs import MansionConfig
from smec_liftsim.person_generators import PersonGeneratorBase

from offline_tools.generate_dataset import generate_dataset_from_csv_to_pipline
import numpy as np


class FixedDataGenerator(PersonGeneratorBase):
    """
    Fetch data from disk and generate person data.
    """
    def __init__(self, data_file='../4Ele16FloorUpPeakFlowUpPeakMode1_elvx.csv', data_of_section='', data_dir=None, file_begin_idx=None):
        super().__init__()
        # print(data_file, data_dir, file_begin_idx)
        self._cur_id = 0
        self.data_file = data_file
        self.data_of_section = data_of_section
        self.data_dir = data_dir
        self.data_files = []
        self.file_idx= 0
        if self.data_dir is not None:
            for file in os.listdir(self.data_dir):
                if file.endswith('.csv'):
                    self.data_files.append(os.path.join(self.data_dir, file))
            if len(self.data_files) > 0:
                if file_begin_idx is not None:
                    self.file_idx = file_begin_idx % len(self.data_files)
                self.data_file = self.data_files[self.file_idx]

        # generate dataset
        self.data, self.used_ele = generate_dataset_from_csv_to_pipline(self.data_file, data_of_section=self.data_of_section)
        self.next_person = None if self.data.empty() else self.data.get()
        self.done = False
        self.total_person_num = self.data.qsize()

    def reset(self):
        # next file
        if len(self.data_files) > 0:
            self.file_idx = (self.file_idx + 1) % len(self.data_files)
            self.data_file = self.data_files[self.file_idx]
            # print(self.data_file)

        self._cur_id = 0
        self.data, self.used_ele = generate_dataset_from_csv_to_pipline(self.data_file, data_of_section=self.data_of_section)
        # print(list(self.data.queue))
        self.next_person = None if self.data.empty() else self.data.get()
        self.done = False
        self.total_person_num = self.data.qsize()

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

