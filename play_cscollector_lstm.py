import random
import time
import numpy as np
from environment import Env
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Model
from keras.layers import Dense, Flatten, Input, LSTM
from keras.layers.convolutional import Conv2D
from keras.layers.wrappers import TimeDistributed

global episode
episode = 0
EPISODES = 8000000

class TestAgent:
    def __init__(self, action_size):
        self.state_size = (20, 84, 84, 1)
        self.action_size = action_size

        self.discount_factor = 0.99
        self.no_op_steps = 30

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
    agent = TestAgent(action_size = 5)
    agent.load_model("./save_model/cscollector_a3c_lstm_actor.h5")

    step = 0

    while episode < EPISODES:
        tick = time.clock()

        done = False

        score = 0
        observe = env.reset()
        next_observe = observe

        for _ in range(random.randint(1, 20)):
            observe = next_observe
            _, _ = env.step(0)
            next_observe = env.render()

            dt = time.clock() - tick
            time.sleep((0.05 - dt) if (0.05 - dt) > 0 else 0)
            tick = time.clock()

        state = pre_processing(observe)
        history = np.stack((state for _ in range(20)), axis = 2)
        history = np.reshape([history], (20, 84, 84, 1))

        while not done:
            dt = time.clock() - tick

            step += 1
            observe = next_observe

            action = agent.get_action(np.array([history]))

            reward, done = env.step(action)
            next_observe = env.render()

            time.sleep((0.05 - dt) if (0.05 - dt) > 0 else 0)
            tick = time.clock()

            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:19, :, :, :], axis = 0)

            score += reward

            # if done, plot the score over episodes and reset the history
            if done:
                history = np.stack(
                    (next_state for _ in range(20)), axis=2)
                history = np.reshape([history], (20, 84, 84, 1))
                episode += 1
                print("episode:", episode, "  score:", score, "  step:", step)
                step = 0
            else:
                history = next_history