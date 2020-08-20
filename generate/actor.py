import numpy as np

class Actor(object):
    def __init__(self, env):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.init_x = 0.0
        self.init_y = 0.0
        self.record = False
        self.env = env
        self.rec_loc_inited = False
        self.t = 0
        self.grasp = 1
        self.gain = 0.05
        self.momentum = 0.5

    def set_values(self,value):
        print("value", value)
        if value == 80:
            self.y += self.gain
            self.x *= self.momentum
            self.z *= self.momentum
        elif value == 59:
            self.y -= self.gain
            self.x *= self.momentum
            self.z *= self.momentum
        elif value == 39:
            self.x += self.gain
            self.y *= self.momentum
            self.z *= self.momentum
        elif value == 76:
            self.x -= self.gain
            self.y *= self.momentum
            self.z *= self.momentum
        elif value == 81:
            self.z += self.gain
            self.x *= self.momentum
            self.y *= self.momentum
        elif value == 87:
            self.z -= self.gain
            self.x *= self.momentum
            self.y *= self.momentum
        elif value == 91:
            self.record = True
            self.rec_loc_inited = False
            print("Recording turned on")
        elif value == 79:
            self.env.reset()
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0
            self.init_x = 0.0
            self.init_y = 0.0
            self.t = 0.0
            self.grasp = 1
            self.record = False
        elif value == 65:
            self.grasp = -1
        

    def get_action(self, obs):
        print('[',self.x,',',self.y,',', self.z,'],')
        return [self.x, self.y, self.z], np.zeros(12), {}

    def reset(self):
        pass


    def on_move(self,x, y):
        if not self.rec_loc_inited:
            self.init_x = x
            self.init_y = y
            self.rec_loc_inited = True
        if self.record:
            self.x = (-self.init_x +x)/300
            self.y = (-y + self.init_y)/300

    def on_click(self,x, y, button, pressed):
        self.record = False
        self.z = -1.0
        self.x = 0
        self.y = 0

    def on_scroll(self,x, y, dx, dy):
        pass