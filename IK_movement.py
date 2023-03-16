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
    fetch.generate_mugs(random_number=False) #generate one mug
    
    fetch.move_to_mug()
    # fetch.set_arm_velocity([0,0,0,0,0,0,0])
    # print('initial config', get_joint_angles())

    for i in range(500):
        fetch.get_image(True)
        p.stepSimulation()
        time.sleep(1./240.)

    fetch.push_mug()
    for i in range(500):
        fetch.get_image(True)
        p.stepSimulation()
        time.sleep(1./240.)

    print('end pos', p.getLinkState(fetch.fetch, 17)[0])
    time.sleep(10)
    p.disconnect()

