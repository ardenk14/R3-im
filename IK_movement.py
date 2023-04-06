import pybullet as p
import pybullet_data
import time
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
from sim_utils import SymFetch
import torch
from r3m import load_r3m

step_dtype = np.dtype([('r3m1', np.float32, 2048),
                        ('q1', np.float32, 7),
                        ('qdot1', np.float32, 7),
                        ('g1', np.bool_, 1),
                        ('x1', np.float32, 3),
                        ('x2', np.float32, 3),
                        ('q2', np.float32, 7),
                        ('g2', np.bool_, 1),
                        ('qdot2', np.float32, 7),
                        ('r3m2', np.float32, 2048)])

def collect_before_data(i, fetch: SymFetch, r3m, data):
    #collect images and state
    im = torch.tensor(fetch.get_image(True))
    data[i]['r3m1'] = r3m(im.permute(2,0,1).reshape(-1, 3, 224, 224)).cpu().numpy()
    data[i]['q1'] = fetch.get_joint_angles()
    data[i]['qdot1'] = fetch.get_joint_vel()
    data[i]['x1'] = fetch.get_ee_pos()
    data[i]['g1'] = fetch.gripper_open

def collect_after_data(i, fetch: SymFetch, r3m, data):
    data[i]['x2'] = fetch.get_ee_pos()
    data[i]['q2'] = fetch.get_joint_angles()
    data[i]['qdot2'] = fetch.get_joint_vel()
    im = torch.tensor(fetch.get_image(True))
    data[i]['r3m2'] = r3m(im.permute(2,0,1).reshape(-1, 3, 224, 224)).cpu().numpy()
    data[i]['g2'] = fetch.gripper_open

def step_sim(i, fps, fetch, data):
    collect_before_data(i, fetch, r3m, data)
    for _ in range(int(240/fps)):
        p.stepSimulation()
        time.sleep(1/240)
    collect_after_data(i, fetch, r3m, data)
    i = i+1
    return i

if __name__ == '__main__':
    with torch.no_grad():
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        r3m = load_r3m("resnet50")
        r3m.to(device)
        r3m.eval()

        fps = 10
        n_samples = 80
        j = 10
        while j < 20:
            i = 0
            fetch = SymFetch(gui=True, random_init=False)
            fetch.generate_blocks(random_number=False, random_color=False, random_pos=False) #generate one block
            
            data = np.zeros(n_samples, dtype=step_dtype)
            # move above the block
            jitter = np.hstack((np.random.uniform(-0.08, 0.08, 2), np.random.uniform(-0.02, 0.02)))
            for _ in range(40):
                fetch.move_to_block(move_above=True, jitter=jitter)
                i = step_sim(i, fps, fetch, data)
            
            # open gripper
            fetch.set_gripper(open=True)
            for _ in range(10):
                fetch.move_to_block(move_above=True)
                i = step_sim(i, fps, fetch, data)

            # lower gripper
            for _ in range(10):
                fetch.move_to_block()
                i = step_sim(i, fps, fetch, data)
            
            # close gripper
            fetch.set_gripper(open=False)
            for _ in range(10):
                i = step_sim(i, fps, fetch, data)

            pos = fetch.get_ee_pos()
            pos = np.array(pos) + [0.0, 0.0, 0.1]

            # lift the block
            for _ in range(10):
                fetch.move_to(pos)
                i = step_sim(i, fps, fetch, data)


            block_pos = p.getBasePositionAndOrientation(fetch.blockIds[0], 0)[0]
            if block_pos[2] > 0.4:
                np.savez_compressed('simple_data{}'.format(j), data=data)
                j += 1
                print('\n\n-------------collected file', j, '--------------')
            else:
                print('\n\n failed')
            time.sleep(3)
            p.disconnect()

