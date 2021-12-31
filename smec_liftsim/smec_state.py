import torch
import numpy as np


class SmecState:
    def __init__(self, up_wait_num=None, down_wait_num=None, loading=None, location=None):
        self.upload_wait_nums = up_wait_num
        self.download_wait_nums = down_wait_num
        self.elevator_loading_maps = loading
        self.elevator_location_maps = location

    def make_tensor(self):
        if type(self.upload_wait_nums) != torch.Tensor:
            self.upload_wait_nums = torch.tensor(self.upload_wait_nums)
            self.download_wait_nums = torch.tensor(self.download_wait_nums)
            self.elevator_loading_maps = torch.tensor(self.elevator_loading_maps)
            self.elevator_location_maps = torch.tensor(self.elevator_location_maps)

    def copy_(self, state):
        if type(self.upload_wait_nums) != torch.Tensor:
            self.upload_wait_nums = torch.tensor(state.upload_wait_nums)
            self.download_wait_nums = torch.tensor(state.download_wait_nums)
            self.elevator_loading_maps = torch.tensor(state.elevator_loading_maps)
            self.elevator_location_maps = torch.tensor(state.elevator_location_maps)
        state.make_tensor()
        self.upload_wait_nums.copy_(state.upload_wait_nums)
        self.download_wait_nums.copy_(state.download_wait_nums)
        self.elevator_loading_maps.copy_(state.elevator_loading_maps)
        self.elevator_location_maps.copy_(state.elevator_location_maps)

    def to(self, dev):
        if self.upload_wait_nums is None:
            return
        self.make_tensor()
        self.upload_wait_nums = self.upload_wait_nums.to(dev)
        self.download_wait_nums = self.download_wait_nums.to(dev)
        self.elevator_loading_maps = self.elevator_loading_maps.to(dev)
        self.elevator_location_maps = self.elevator_location_maps.to(dev)

    @property
    def device(self):
        return self.upload_wait_nums.device
