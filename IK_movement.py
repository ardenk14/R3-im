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
    fetch.generate_blocks(random_number=False, random_color=True) #generate one block
    
    for _ in range(20):
        fetch.move_to_block(move_above=True)
        for i in range(50):
            if i%16==0:
                fetch.get_image(True)
            p.stepSimulation()
            time.sleep(1./240.)
    
    fetch.set_gripper(open=True)
    for i in range(100):
        if i%16==0:
            fetch.get_image(True)
        p.stepSimulation()
        time.sleep(1./240.)

    
    for _ in range(3):
        fetch.move_to_block()
        for i in range(50):
            if i%16==0:
                fetch.get_image(True)
            p.stepSimulation()
            time.sleep(1./240.)
    
    #pick up the block
    fetch.set_gripper(open=False)
    for i in range(100):
        if i%16==0:
            fetch.get_image(True)
        p.stepSimulation()
        time.sleep(1./240.)

    pos = fetch.get_ee_pos()
    pos = np.array(pos) + [0.0, 0.0, 0.1]

    for _ in range(10):
        fetch.move_to(pos)
        for i in range(50):
            if i%16==0:
                fetch.get_image(True)
            p.stepSimulation()
            time.sleep(1./240.)

    print('end pos', p.getLinkState(fetch.fetch, 17)[0])
    time.sleep(10)
    p.disconnect()

