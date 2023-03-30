import pybullet as p
import pybullet_data
import time
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
from sim_utils import SymFetch

def get_joint_angles():
    states = p.getJointStates(fetch.fetch, [i for i in range(17)])
    q = [state[0] for state in states]
    return q

if __name__ == '__main__':
    fetch = SymFetch()
    fetch.generate_mugs(random_number=False, random_color=True) #generate one mug
    
    for _ in range(20):
        fetch.move_to_mug(move_above=True)
        for i in range(50):
            if i%16==0:
                fetch.get_image(True)
            p.stepSimulation()
            time.sleep(1./240.)
        print(fetch.get_ee_pos())
    
    for _ in range(3):
        fetch.move_to_mug()
        for i in range(50):
            if i%16==0:
                fetch.get_image(True)
            p.stepSimulation()
            time.sleep(1./240.)
        print(fetch.get_ee_pos())
    # mug_x_lim = [0.7, 1.0] #limits for mug position
    # mug_y_lim = [-.5, .5]
    # for pt in [[0.7, -0.5, 0.6], [0.7, 0.5, 0.6], [1.0, -0.5, 0.6], [1.0, 0.5, 0.6]]:
    #     fetch.move_to(pt)
    #     for i in range(1000):
    #         if i%16==0:
    #             fetch.get_image(True)
    #         p.stepSimulation()
    #         time.sleep(1./240.)
    #     print('end pos', p.getLinkState(fetch.fetch, 17)[0])

    fetch.push_mug()
    for i in range(500):
        if i%16==0:
            fetch.get_image(True)
        p.stepSimulation()
        time.sleep(1./240.)

    print('end pos', p.getLinkState(fetch.fetch, 17)[0])
    time.sleep(10)
    p.disconnect()

