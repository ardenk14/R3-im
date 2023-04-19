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
import tqdm
from multiprocessing import Pool
import multiprocessing as mp
from itertools import repeat


def collect_before_data(i, fetch: SymFetch, r3m, data, last_action):
    #collect images and state
    im = torch.tensor(fetch.get_image(True))
    data[i]['r3m1'] = r3m(im.permute(2,0,1).reshape(-1, 3, 224, 224)).cpu().numpy()
    data[i]['q1'] = fetch.get_joint_angles()
    data[i]['last_action'] = last_action
    data[i]['g1'] = fetch.get_gripper_state()

def collect_after_data(i, fetch: SymFetch, r3m, data):
    data[i]['q2'] = fetch.get_joint_angles()
    im = torch.tensor(fetch.get_image(True))
    data[i]['r3m2'] = r3m(im.permute(2,0,1).reshape(-1, 3, 224, 224)).cpu().numpy()
    data[i]['g2'] = fetch.gripper_open

def step_sim(i, fps, fetch, data, r3m, last_action):
    collect_before_data(i, fetch, r3m, data, last_action)
    for _ in range(int(240/fps)):
        fetch.stepSimulation()
        # time.sleep(1/240)
    collect_after_data(i, fetch, r3m, data)
    i = i+1
    return i

def pick_place(j, r3m):
    with torch.no_grad():
        step_dtype = np.dtype([('r3m1', np.float32, 2048),
                                ('q1', np.float32, 7),
                                ('g1', np.float32, 3),
                                ('last_action', np.float32, 7),
                                ('q2', np.float32, 7),
                                ('g2', np.bool_, 1),
                                ('qdot2', np.float32, 7),
                                ('r3m2', np.float32, 2048)])

        fps = 10
        n_samples = 1000
        i = 0
        fetch = SymFetch(gui=True, random_init=True)
        fetch.generate_blocks(random_number=True, random_color=True, random_pos=True) #generate many blocks
        
        data = np.zeros(n_samples, dtype=step_dtype)
        last_action = np.zeros(7)
        for _ in range(20):
            dist = 1
            k = 0
            x = np.random.uniform(0.6, 0.8)
            y = np.random.uniform(-0.4, 0.4)
            z = np.random.uniform(0.4, 0.6)
            pos = np.array([x, y, z])
            while dist > 0.1 and k < 40:
                dist = fetch.move_to(pos)
                q = fetch.get_joint_angles()
                i = step_sim(i, fps, fetch, data, r3m, last_action)
                last_action = fetch.get_joint_angles() - q
                # print(dist, fetch.get_gripper_state())
                k += 1

            action_size = 0.3
            for rand_action in range(3):
                action = np.random.uniform(-action_size, action_size, 7)
                for rand_step in range(5):
                    q = fetch.get_joint_angles()
                    fetch.set_joint_angles(q + action)
                    i = step_sim(i, fps, fetch, data, r3m, last_action)
                    last_action = fetch.get_joint_angles() - q

        fetch.disconnect()


    data = data[:i]
    print(i)
    np.savez_compressed('random_walk{}'.format(j), data=data)
    print('\n\n-------------collected file', j, '--------------')

if __name__=="__main__":
    mp.set_start_method('spawn', force=True)
    

    with torch.no_grad():
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        r3m = load_r3m("resnet50")
        r3m.to(device)
        r3m.eval()
        r3m.share_memory()
        for i in range(31,40,2):
            processes = []
            for j in range(2):
                pr = mp.Process(target=pick_place, args=(i+j, r3m))
                pr.start()
                processes.append(pr)
            for pr in processes:
                pr.join()
    

