import numpy as np
from sim_utils import SymFetch
import time
import pybullet as p

data = np.load('data/simple/simple_data6.npz')['data']

fetch = SymFetch(gui=True, random_init=False)
fetch.generate_blocks(random_number=False, random_color=False, random_pos=False) #generate one block

fps = 10
for step in data:
    fetch.set_joint_angles(step['q2'])
    fetch.set_gripper(step['g2'])
    print(step['g2'])
    for _ in range(int(240/fps)):
        p.stepSimulation()
        time.sleep(1/240)
