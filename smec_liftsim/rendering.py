#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pyglet
import os
import random
from smec_liftsim.utils import EPSILON

from PIL import Image
import glob

pyglet.resource.path = [(os.path.dirname(__file__) + '/resources')]
pyglet.resource.reindex()
scale_y = 12.5


class Render(pyglet.window.Window):
    def __init__(self, shared):
        self.shared_mansion = shared
        self.floor_height = self.shared_mansion.attribute.FloorHeight
        self.elevator_num = self.shared_mansion.attribute.ElevatorNumber
        self.num_floor = self.shared_mansion.attribute.NumberOfFloor

        self.screen_x = int(self.elevator_num * 50 + 300)
        self.screen_y = int(self.num_floor * scale_y * self.floor_height + 90)
        super(Render, self).__init__(width=self.screen_x, height=self.screen_y)
        self.create_window()

        self.frame = []
        self.image_count = 0
        self.gif_count = 0
        if not os.path.exists('./image_buffer'):
            os.makedirs('./image_buffer')

        if not os.path.exists('./animation_buffer'):
            os.makedirs('./animation_buffer')

    def create_window(self):
        self.set_size(self.screen_x, self.screen_y)
        self.set_visible()

        self.load_images()
        self.init_batch()

    def center_image(self, image):
        image.anchor_x = image.width // 2
        image.anchor_y = image.height // 2

    def load_images(self):
        self.man_image_1 = pyglet.resource.image('1.png')
        self.man_image_2 = pyglet.resource.image('2.png')
        self.man_image_3 = pyglet.resource.image('3.png')
        self.man_image_4 = pyglet.resource.image('4.png')
        self.human_image = [self.man_image_1, self.man_image_2, self.man_image_3, self.man_image_4]
        self.background = pyglet.resource.image("background.png")
        self.line = pyglet.resource.image("line.png")
        self.up = pyglet.resource.image("up.png")
        self.up_open_close = pyglet.resource.image("up_open_close.png")
        self.up_full_open = pyglet.resource.image("up_full_open.png")
        self.down = pyglet.resource.image("down.png")
        self.down_open_close = pyglet.resource.image("down_open_close.png")
        self.down_full_open = pyglet.resource.image("down_full_open.png")
        self.steady = pyglet.resource.image("steady.png")
        self.steady_open_close = pyglet.resource.image("steady_open_close.png")
        self.steady_full_open = pyglet.resource.image("steady_full_open.png")

        self.line.width, self.line.height = self.screen_x, 3
        self.center_image(self.line)

        self.background.width, self.background.height = self.screen_x, self.screen_y
        self.center_image(self.background)
        self.background = pyglet.sprite.Sprite(img=self.background, x=self.screen_x // 2, y=self.screen_y // 2)

        for image in self.human_image:
            image.width, image.height = 20, 38
            self.center_image(image)

        self.up.width, self.up.height = 35, 45
        self.center_image(self.up)

        self.up_open_close.width, self.up_open_close.height = 35, 45
        self.center_image(self.up_open_close)

        self.up_full_open.width, self.up_full_open.height = 35, 45
        self.center_image(self.up_full_open)

        self.down.width, self.down.height = 35, 45
        self.center_image(self.down)

        self.down_open_close.width, self.down_open_close.height = 35, 45
        self.center_image(self.down_open_close)

        self.down_full_open.width, self.down_full_open.height = 35, 45
        self.center_image(self.down_full_open)

        self.steady.width, self.steady.height = 35, 50
        self.center_image(self.steady)

        self.steady_open_close.width, self.steady_open_close.height = 35, 50
        self.center_image(self.steady_open_close)

        self.steady_full_open.width, self.steady_full_open.height = 35, 50
        self.center_image(self.steady_full_open)
        self.time_cnt_label = pyglet.text.Label(text="00:00", font_size=12, x=100, y=self.screen_y - 35,
                                                anchor_x='center')
        self.up_label = pyglet.text.Label(text="Waiting Up", font_size=12, x=100, y=self.screen_y - 35,
                                          anchor_x='center')
        self.down_label = pyglet.text.Label(text="Waiting Down", font_size=12, x=self.screen_x - 100,
                                            y=self.screen_y - 35, anchor_x='center')
        self.level_label = pyglet.text.Label(text="Elevator Simulator", font_size=12, x=self.screen_x // 2,
                                             y=self.screen_y - 15, anchor_x='center')

        self.test_img = pyglet.sprite.Sprite(img=self.up_full_open, x=25, y=22.5)

    def init_batch(self):
        """
        line_batch: lines to separate floors, which can be initialized here as it remains unchanged
        waiting_people_batch: people on the two sides
        elevator_batch: including elevator (square) and passengers (circle)
        """
        self.line_batch = pyglet.graphics.Batch()
        self.level_num_batch = pyglet.graphics.Batch()
        self.waiting_people_batch = pyglet.graphics.Batch()
        self.elevator_batch = list()
        self.carcall_batch = list()
        self.line_ele = list()
        self.level_num_ele = list()
        self.waiting_people_ele = list()
        self.elevator_ele = list()
        self.carcall_ele = list()
        self.right_list = [[] for i in range(self.num_floor)]
        self.left_list = [[] for i in range(self.num_floor)]
        for i in range(self.elevator_num):
            self.elevator_batch.append(pyglet.graphics.Batch())
            self.carcall_batch.append(pyglet.graphics.Batch())

        for i in range(self.num_floor):
            self.line_ele.append(
                pyglet.sprite.Sprite(img=self.line, x=self.screen_x // 2, y=scale_y * self.floor_height * (i + 1),
                                     batch=self.line_batch))
            self.level_num_ele.append(
                pyglet.text.Label(text=f"{i + 1}", font_size=15, x=25,
                                  y=scale_y * self.floor_height * i + 5, anchor_x='center', batch=self.level_num_batch)
            )

    def update(self):
        # update time
        time_cnt = self.shared_mansion._config.raw_time
        self.time_cnt_label.delete()
        self.time_cnt_label = pyglet.text.Label(text="{}:{:.2f}".format(int(time_cnt // 60), time_cnt % 60),
                                                font_size=12, x=10, y=self.screen_y - 70,
                                                anchor_x='left')

        # update waiting_people_batch
        waiting_up, waiting_down = self.shared_mansion.waiting_queue
        for ele in self.waiting_people_ele:
            ele.delete()
        self.waiting_people_ele = []
        for i in range(self.num_floor):
            # left side, waiting up people
            if len(waiting_up[i]) > 9:
                self.waiting_people_ele.append(
                    pyglet.sprite.Sprite(img=self.man_image_1, x=100, y=scale_y * self.floor_height * i + 20,
                                         batch=self.waiting_people_batch))
                self.waiting_people_ele.append(pyglet.text.Label(text="x {}".format(len(waiting_up[i])), font_size=8,
                                                                 x=130, y=scale_y * self.floor_height * i + 15,
                                                                 anchor_x='center', batch=self.waiting_people_batch))
            else:
                while len(self.left_list[i]) < len(waiting_up[i]):
                    self.left_list[i].append(random.randint(0, 3))
                while len(self.left_list[i]) > len(waiting_up[i]):
                    self.left_list[i].pop(0)
                for j in range(len(waiting_up[i])):
                    self.waiting_people_ele.append(
                        pyglet.sprite.Sprite(img=self.human_image[self.left_list[i][j]], x=140 - 15 * j,
                                             y=scale_y * self.floor_height * i + 22, batch=self.waiting_people_batch))
            if self.shared_mansion.up_served_call[i] != -1:
                self.waiting_people_ele.append(
                    pyglet.text.Label(text=str(self.shared_mansion.up_served_call[i]), font_size=20, x=95,
                                      y=scale_y * self.floor_height * i + 15,
                                      anchor_x='center', batch=self.waiting_people_batch))
            # right side, waiting down people
            if len(waiting_down[i]) > 9:
                self.waiting_people_ele.append(pyglet.sprite.Sprite(img=self.man_image_1, x=self.screen_x - 125,
                                                                    y=scale_y * self.floor_height * i + 20,
                                                                    batch=self.waiting_people_batch))
                self.waiting_people_ele.append(pyglet.text.Label(text="x {}".format(len(waiting_down[i])), font_size=8,
                                                                 x=self.screen_x - 95,
                                                                 y=scale_y * self.floor_height * i + 15,
                                                                 anchor_x='center',
                                                                 batch=self.waiting_people_batch))
            else:
                while len(self.right_list[i]) < len(waiting_down[i]):
                    self.right_list[i].append(random.randint(0, 3))
                while len(self.right_list[i]) > len(waiting_down[i]):
                    self.right_list[i].pop(0)
                for j in range(len(waiting_down[i])):
                    self.waiting_people_ele.append(pyglet.sprite.Sprite(img=self.human_image[self.right_list[i][j]],
                                                                        x=self.screen_x - 140 + 15 * j,
                                                                        y=scale_y * self.floor_height * i + 22,
                                                                        batch=self.waiting_people_batch))
            if self.shared_mansion.dn_served_call[i] != -1:
                self.waiting_people_ele.append(pyglet.text.Label(text=str(self.shared_mansion.dn_served_call[i]),
                                                                 font_size=20, x=self.screen_x - 95,
                                                                 y=scale_y * self.floor_height * i + 15,
                                                                 anchor_x='center',
                                                                 batch=self.waiting_people_batch))
        # update elevator_batch
        for ele in self.elevator_ele:
            ele.delete()
        self.elevator_ele = []
        height = self.up_full_open.height / 2
        for i in range(self.elevator_num):
            self.elevator_floor = self.shared_mansion.state.ElevatorStates[i].Floor
            self.elevator_pos = self.shared_mansion.state.ElevatorStates[i].Position
            img_x = 175 + i * 50
            img_y = self.elevator_pos * scale_y + height
            if self.shared_mansion._elevators[i]._service_direction == 1:
                if self.shared_mansion._elevators[i]._door_open_rate > 1.0 - EPSILON:
                    self.elevator_ele.append(pyglet.sprite.Sprite(img=self.up_full_open, x=img_x,
                                                                  y=img_y,
                                                                  batch=self.elevator_batch[i]))
                elif self.shared_mansion._elevators[i]._door_open_rate < EPSILON:
                    self.elevator_ele.append(pyglet.sprite.Sprite(img=self.up, x=img_x,
                                                                  y=img_y,
                                                                  batch=self.elevator_batch[i]))
                else:
                    self.elevator_ele.append(pyglet.sprite.Sprite(img=self.up_open_close, x=img_x,
                                                                  y=img_y,
                                                                  batch=self.elevator_batch[i]))
            elif self.shared_mansion._elevators[i]._service_direction == -1:
                if self.shared_mansion._elevators[i]._door_open_rate > 1.0 - EPSILON:
                    self.elevator_ele.append(pyglet.sprite.Sprite(img=self.down_full_open, x=img_x,
                                                                  y=img_y,
                                                                  batch=self.elevator_batch[i]))
                elif self.shared_mansion._elevators[i]._door_open_rate < EPSILON:
                    self.elevator_ele.append(pyglet.sprite.Sprite(img=self.down, x=img_x,
                                                                  y=img_y,
                                                                  batch=self.elevator_batch[i]))
                else:
                    self.elevator_ele.append(pyglet.sprite.Sprite(img=self.down_open_close, x=img_x,
                                                                  y=img_y,
                                                                  batch=self.elevator_batch[i]))
            else:
                if self.shared_mansion._elevators[i]._door_open_rate > 1.0 - EPSILON:
                    self.elevator_ele.append(pyglet.sprite.Sprite(img=self.steady_full_open, x=img_x,
                                                                  y=img_y,
                                                                  batch=self.elevator_batch[i]))
                elif self.shared_mansion._elevators[i]._door_open_rate < EPSILON:
                    self.elevator_ele.append(pyglet.sprite.Sprite(img=self.steady, x=img_x,
                                                                  y=img_y,
                                                                  batch=self.elevator_batch[i]))
                else:
                    self.elevator_ele.append(pyglet.sprite.Sprite(img=self.steady_open_close, x=img_x,
                                                                  y=img_y,
                                                                  batch=self.elevator_batch[i]))
            # when too many passengers in an elevator, use numeric numbers to show
            self.elevator_ele.append(pyglet.text.Label(text="loading", font_size=8,
                                                       x=img_x, y=self.screen_y - 45, anchor_x='center',
                                                       batch=self.elevator_batch[i]))
            self.elevator_ele.append(
                pyglet.text.Label(text="{} people".format(self.shared_mansion.loaded_people[i]), font_size=8,
                                  x=img_x, y=self.screen_y - 58, anchor_x='center',
                                  batch=self.elevator_batch[i]))
            self.elevator_ele.append(
                pyglet.text.Label(text="{:.2f}".format(self.shared_mansion._elevators[i]._door_open_rate), font_size=8,
                                  x=img_x, y=self.screen_y - 71, anchor_x='center',
                                  batch=self.elevator_batch[i]))
            self.elevator_ele.append(
                pyglet.text.Label(text="{:.2f}".format(self.shared_mansion._elevators[i]._current_velocity),
                                  font_size=8,
                                  x=img_x, y=self.screen_y - 84, anchor_x='center',
                                  batch=self.elevator_batch[i]))

        # update carcall_batch
        for ele in self.carcall_ele:
            ele.delete()
        self.carcall_ele = []
        for i in range(self.elevator_num):
            carcall = self.shared_mansion._elevators[i]._car_call
            for cc in carcall:
                cc_x = 175 + i * 50
                cc_y = scale_y * self.floor_height * cc + 5
                self.carcall_ele.append(pyglet.text.Label(text='*',
                                                          font_size=20, x=cc_x,
                                                          color=(0,0,0,255),
                                                          y=cc_y,
                                                          anchor_x='center',
                                                          batch=self.carcall_batch[i]))

        # pyglet.gl.get_current_context().set_current()  # this is deprecated api
        pyglet.gl.current_context.set_current()

    def view(self):
        self.clear()
        self.update()

        self.dispatch_events()
        self.background.draw()
        self.line_batch.draw()
        self.level_num_batch.draw()
        self.level_label.draw()
        self.up_label.draw()
        self.down_label.draw()
        self.time_cnt_label.draw()
        self.waiting_people_batch.draw()
        for i in range(self.elevator_num):
            self.elevator_batch[i].draw()
            self.carcall_batch[i].draw()
        '''
        # save the current window image
        pyglet.image.get_buffer_manager().get_color_buffer().save('./image_buffer/image{}.png'.format(str(self.image_count)))
        image = glob.glob("./image_buffer/image{}.png".format(str(self.image_count)))
        new_frame = Image.open(image[0])
        self.frame.append(new_frame)
        self.image_count += 1
        '''
        # print the window
        self.flip()
        '''
        if self.image_count == 1000:
            # combine images in self.frame to gif and save
            self.frame[0].save('./animation_buffer/animation{}.gif'.format(str(self.gif_count)), 
                               format='GIF', append_images=self.frame[1:], 
                               save_all=True, duration=10, loop=0)
            self.frame = []
            self.image_count = 0
            self.gif_count += 1
        '''
