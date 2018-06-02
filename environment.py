import sys, pygame
import random
import time
import numpy as np
import math

# Predefined transform functions
def rotate(x, angle):
    return np.matmul(np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]), x)

def isColla(a, b, dist):
    if np.linalg.norm(a-b) < 2.0*dist:
        return True
    else:
#        print(np.linalg.norm(a-b))
        return False

class Env():
    # initialize
    def __init__(self, mode = True, show = True):
        pygame.init()
        self.mode = mode

        self.size = self.width, self.height = 640, 640
        self.mapsize = 560, 560
        self.screen = None
        if self.mode or show:
            self.screen = pygame.display.set_mode(self.size)
        else:
            self.screen = pygame.Surface(self.size)

        self.speed = 15
        self.omega = 8
        self.direction = 0
        self.itemsize = 22
        self.linewidth = 16
        self.tpan = .01
        self.wpan = .5
        self.scoreboard = [60., -80.]
        self.margin = self.itemsize + self.linewidth/2
        self.univec = np.array([1., 0.])
        self.cret = 1
        self.itemnum = [3, 3]
        self.itemtype = [0 for _ in range(self.itemnum[0])] + [1 for _ in range(self.itemnum[1])]

        self.score = 0.
        self.stepcnt = 0
        self.icnt = 0
        self.genitems()

    def reset(self):
        self.score = 0.
        self.stepcnt = 0
        self.icnt = 0
        self.genitems()

        return self.render(show = False)

    def genitems(self):
        self.jo = np.array([self.mapsize[0] / 2, self.mapsize[1] / 2])
        self.items = []
        for i in range(self.itemnum[0] + self.itemnum[1]):
            collide = True
            newitem = []
            while collide:
                collide = False
                newitem = [int(self.margin + random.random() * (self.mapsize[0] - 2 * self.margin)),
                                int(self.margin + random.random() * (self.mapsize[1] - 2 * self.margin)),
                                self.itemtype[i]]
                if isColla(self.jo, np.array(newitem[0:2]), self.itemsize):
                    collide = True
                for j in range(i):
                    if isColla(np.array(self.items[j][0:2]), np.array(newitem[0:2]), self.itemsize):
                        collide = True
                        break
            self.items.append(newitem)
        self.items = np.array(self.items)

    # loop
    def event_handle(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
        return 0

    def step(self, action = 0):
        #global direction, speed, omega, jo, items, score
        reward = 0.
        self.stepcnt += 1
        prevjo = np.copy(self.jo)

        keystate = pygame.key.get_pressed()
        if keystate[pygame.K_UP] if self.mode else action == 1:
            self.jo += (self.speed*rotate(self.univec, math.pi*self.direction/180)).astype(int)
        if keystate[pygame.K_RIGHT] if self.mode else action == 2:
            self.direction = (self.direction + self.omega) % 360
        if keystate[pygame.K_DOWN] if self.mode else action == 3:
            self.jo -= (self.speed*rotate(self.univec, math.pi*self.direction/180)).astype(int)
        if keystate[pygame.K_LEFT] if self.mode else action == 4:
            self.direction = (self.direction - self.omega) % 360

        if self.jo[0] < self.margin or self.jo[0] > self.mapsize[0] - self.margin or self.jo[1] < self.margin or \
                self.jo[1] > self.mapsize[1] - self.margin:
            self.jo = prevjo
            reward -= self.wpan
            self.score -= self.wpan

        elif np.linalg.norm(self.jo - prevjo) < self.speed / 2:
        	reward -= 2*self.tpan
        	self.score -= 2*self.tpan

        newarr = None
        isUpdated = False

        for idx in range(self.items.shape[0]):
            if isColla(self.jo, self.items[idx][0:2], self.itemsize):
                self.score += self.scoreboard[int(self.items[idx][2])]
                reward += self.scoreboard[int(self.items[idx][2])]
                if self.items[idx][2] == 0:
                    self.icnt += 1
                newarr = np.delete(self.items, idx, 0)
                isUpdated = True

        if isUpdated:
            self.items = newarr

        #if self.stepcnt % self.cret == 0:
        if action == 0:
            self.score -= self.tpan
            reward -= self.tpan

        if action == 3:
        	self.score -= self.tpan
        	reward -= self.tpan/2

        return reward, (self.itemnum[0] == self.icnt)


    def render(self, show = True):
        #global jo
        self.screen.fill((0, 0, 0))

        for item in self.items:
            itemcor = rotate(np.array([item[0] - self.jo[0], item[1] - self.jo[1]]), -math.pi*(self.direction + 90)/180)
            if item[2] == 0.:
                pygame.draw.circle(self.screen, (255*item[2], 255*(1-item[2]), 255*item[2]), [int(itemcor[0] + self.width/2), int(itemcor[1] + self.height/2)], self.itemsize)
            else:
                plist = self.itemsize * math.sqrt(2) * np.array([[0., -1.], [math.sqrt(3)/2., .5], [-math.sqrt(3)/2., .5]])
                pygame.draw.polygon(self.screen, (255*item[2], 255*(1-item[2]), 255*item[2]), (plist + itemcor + np.array(self.size)/2).astype(int).tolist())
            #print(item)

        plist = np.array([
            rotate(np.array([0 - self.jo[0], 0 - self.jo[1]]), -math.pi * (self.direction + 90) / 180),
            rotate(np.array([self.mapsize[0] - self.jo[0], 0 - self.jo[1]]), -math.pi * (self.direction + 90) / 180),
            rotate(np.array([self.mapsize[0] - self.jo[0], self.mapsize[1] - self.jo[1]]), -math.pi * (self.direction + 90) / 180),
            rotate(np.array([0 - self.jo[0], self.mapsize[1] - self.jo[1]]), -math.pi * (self.direction + 90) / 180)])\
                + np.array([[self.width/2, self.height/2] for _ in range(4)])

        pygame.draw.polygon(self.screen, (255, 255, 255), plist.astype(int).tolist(), self.linewidth)
        pygame.draw.rect(self.screen, (255, 255, 0), pygame.Rect(int(self.width / 2 - self.itemsize), int(self.height / 2- self.itemsize), 2*self.itemsize, 2*self.itemsize))

#        print(self.jo)
#        print("Direction : " + str(self.direction))
#        print(self.score)
        if show:
            pygame.display.flip()

        #retval = pygame.surfarray.array3d(self.screen)

        return pygame.surfarray.array3d(self.screen)

    def update(self):
        if self.event_handle() == None:
            return False
        self.step()
        self.render()
        return True

if __name__ == "__main__":
    game = Env()
    tick = time.clock()
    # LOOOOOOOOP
    while game.update():
        dt = time.clock() - tick
        time.sleep((0.05 - dt) if (0.05 - dt) > 0 else 0)
        tick = time.clock()
    # finalize