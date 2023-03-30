import pybullet as p
import pybullet_data
import time
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
from sim_utils import SymFetch
import pygame

fetch = SymFetch(gui=True)
fetch.generate_mugs(random_color=True)

pygame.init()
pygame.joystick.init()
joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]

pos = fetch.get_ee_pos()
# fetch.move_to(pos)
# print("here")
for _ in range(300):
    p.stepSimulation()

try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
        
        d1 = -joysticks[0].get_axis(0)
        d2 = joysticks[0].get_axis(1)
        d3 = (joysticks[0].get_axis(5) + 1)/2 - (joysticks[0].get_axis(2) + 1)/2

        d4 = joysticks[0].get_axis(3)
        d5 = joysticks[0].get_axis(4)
        scale = 0.5

        cmd = np.array([d1, d2, d3, d4, d5])*scale
        vel_cmd = np.zeros(7)

        if np.linalg.norm(cmd) > scale/2:
            vel_cmd[:len(cmd)] = cmd

        fetch.set_arm_velocity(vel_cmd)


        for i in range(50):
            if i%16==0:
                fetch.get_image(True)
            p.stepSimulation()
            time.sleep(1./240.)


except KeyboardInterrupt as e:
    pygame.quit()
    print("exit")