import random
import time
import sys, pygame
import numpy as np
import math
import socket

from coordinates import rela_coords

from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Model
from keras.layers import Dense, Flatten, Input, LSTM
from keras.layers.convolutional import Conv2D
from keras.layers.wrappers import TimeDistributed

global episode
episode = 0
EPISODES = 100

# Predefined transform functions
def rotate(x, angle):
    return np.matmul(np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]), x)

def isColla(a, b, dist):
    if np.linalg.norm(a-b) < 2.0*dist:
        return True
    else:
        return False

class Dummy():
    def __init__(self):
        pass

    def send_to_ev3(self, signal = 1):
        print(signal, " was successfully sent to ev3!!")

class Comm():
    def __init__(self, addr, port):
        self.server_address = addr
        self.port = port
        self.size = 1024
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.server_address, port))

        self.sock.listen(1)
        print("Waiting ev3Client...")

        try:
            self.client, self.clientInfo = self.sock.accept()
            print("Connected client:", clientInfo)

        except:
            print("Closing socket")
            self.client.close()
            self.sock.close()

    def send_to_ev3(self, signal = 1):
        try:
            signal = str(signal) + "\n"
            if signal:
                print("Sended: " + signal)
                client.sendall(signal.encode('UTF-8'))

            else:
                print("Disconnected")
                client.close()
                sock.close()
        except:
            print("Closing socket")
            self.client.close()
            self.sock.close()

class Env():
    # initialize
    def __init__(self, mode = True, show = True):
        pygame.init()
        self.ev3 = Comm(addr = "192.168.137.1", port = 8040)
        #self.ev3 = Dummy()
        self.mode = mode

        self.size = self.width, self.height = 640, 640
        self.mapsize = 560, 560
        self.screen = None
        if self.mode or show:
            self.screen = pygame.display.set_mode(self.size)
        else:
            self.screen = pygame.Surface(self.size)

        self.itemsize = 22
        self.linewidth = 16

        self.margin = self.itemsize + self.linewidth/2
        self.univec = np.array([1., 0.])

        self.itemnum = [2, 2]
        self.itemtype = [0 for _ in range(self.itemnum[0])] + [1 for _ in range(self.itemnum[1])]

    def reset(self):
        self.step(0)
        return self.render(show = False)

    def event_handle(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
        return 0

    def step(self, action = 0):
        self.itemnum, mapobj = rela_coords()

        self.itemtype = [0 for _ in range(self.itemnum[0])] + [1 for _ in range(self.itemnum[1])]
        itemtype_ = [[_] for _ in self.itemtype]
        self.items = mapobj[2:]
        #print(self.items.shape)
        #print(self.items)
        try:
            self.items = np.append(self.items, itemtype_, axis = 1)

        except:
            print(self.itemnum)

        # Approached to milk pack
        for i in range(self.itemnum[0]):
            if isColla(self.items[i][:2], np.array([0, 0]), self.itemsize):
                action = 5

        # Communication: Sending message
        self.ev3.send_to_ev3(action)

        mapcenter = np.array([(mapobj[0][0] + mapobj[1][0])/2, (mapobj[0][1] + mapobj[1][1])/2])
        self.mapcoor = np.array([mapobj[0], rotate(mapobj[1]-mapcenter, math.pi/2) + mapcenter, mapobj[1], rotate(mapobj[1]-mapcenter, -math.pi/2) + mapcenter])

        print(self.mapcoor)
        return (self.itemnum[0] == 0)

    def render(self, show = True):
        self.screen.fill((0, 0, 0))

        for item in self.items:
            if item[2] == 0.:
                pygame.draw.circle(self.screen, (255*item[2], 255*(1-item[2]), 255*item[2]), (item[:2] + np.array(self.size)/2).astype(int).tolist(), self.itemsize)
            else:
                plist = self.itemsize * math.sqrt(2) * np.array([[0., -1.], [math.sqrt(3)/2., .5], [-math.sqrt(3)/2., .5]])
                pygame.draw.polygon(self.screen, (255*item[2], 255*(1-item[2]), 255*item[2]), (plist + item[:2] + np.array(self.size)/2).astype(int).tolist())

        pygame.draw.polygon(self.screen, (255, 255, 255), (self.mapcoor + np.array([[self.width/2, self.height/2] for _ in range(4)])).astype(int).tolist(), self.linewidth)
        pygame.draw.rect(self.screen, (255, 255, 0), pygame.Rect(int(self.width / 2 - self.itemsize), int(self.height / 2- self.itemsize), 2*self.itemsize, 2*self.itemsize))

        if show:
            pygame.display.flip()

        return pygame.surfarray.array3d(self.screen)

    def update(self):
        if self.event_handle() == None:
            return False
        self.step()
        self.render()
        return True

class TestAgent:
    def __init__(self, action_size):
        self.state_size = (20, 84, 84, 1)
        self.action_size = action_size

        self.discount_factor = 0.99
        self.no_op_steps = 3

        self.actor, self.critic = self.build_model()

    def build_model(self):
        input = Input(shape=self.state_size)
        conv = TimeDistributed(Conv2D(16, (8, 8), strides=(4, 4), activation='relu'))(input)
        conv = TimeDistributed(Conv2D(32, (4, 4), strides=(2, 2), activation='relu'))(conv)
        conv = TimeDistributed(Flatten())(conv)
        fc = TimeDistributed(Dense(256, activation='relu'))(conv)
        lstm = LSTM(256)(fc)

        policy = Dense(self.action_size, activation='softmax')(lstm)
        value = Dense(1, activation='linear')(lstm)

        actor = Model(inputs=input, outputs=policy)
        critic = Model(inputs=input, outputs=value)

        actor.summary()
        critic.summary()

        return actor, critic

    def get_action(self, history):
        history = np.float32(history / 255.)
        policy = self.actor.predict(history)[0]

        action_index = np.argmax(policy)
        return action_index

    def load_model(self, name):
        self.actor.load_weights(name)

def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe


if __name__ == "__main__":
    tick = time.clock()
    env = Env(mode = False)
    agent = TestAgent(action_size=5)
    agent.load_model("./save_model/cscollector_a3c_lstm_actor.h5")

    waittime = 1.
    step = 0

    while episode < EPISODES:
        tick = time.clock()

        done = False

        observe = env.reset()
        next_observe = observe

        for _ in range(random.randint(1, agent.no_op_steps)):
            observe = next_observe
            _ = env.step(0)
            next_observe = env.render()

            # waiting for waittime second
            dt = time.clock() - tick
            time.sleep((waittime - dt) if (waittime - dt) > 0 else 0)
            tick = time.clock()

        state = pre_processing(observe)
        history = np.stack((state for _ in range(20)), axis=2)
        history = np.reshape([history], (20, 84, 84, 1))

        while not done:
            dt = time.clock() - tick

            step += 1
            observe = next_observe

            action = agent.get_action(np.array([history]))
            print(step, action)
            done = env.step(action)
            next_observe = env.render()

            # waiting for waittime second
            time.sleep((waittime - dt) if (waittime - dt) > 0 else 0)
            tick = time.clock()

            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:19, :, :, :], axis=0)

            # if done, plot the episodes and reset the history
            if done:
                history = np.stack(
                    (next_state for _ in range(20)), axis=2)
                history = np.reshape([history], (20, 84, 84, 1))
                episode += 1
                print("episode:", episode, "  time:", step*waittime)
                step = 0
            else:
                history = next_history