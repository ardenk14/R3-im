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


def collect_before_data(i, fetch: SymFetch, r3m, data):
    #collect images and state
    im = torch.tensor(fetch.get_image(True))
    data[i]['r3m1'] = r3m(im.permute(2,0,1).reshape(-1, 3, 224, 224)).cpu().numpy()
    data[i]['q1'] = fetch.get_joint_angles()
    data[i]['qdot1'] = fetch.get_joint_vel()
    data[i]['x1'] = fetch.get_ee_pos()
    data[i]['g1'] = fetch.get_gripper_state()

def collect_after_data(i, fetch: SymFetch, r3m, data):
    data[i]['x2'] = fetch.get_ee_pos()
    data[i]['q2'] = fetch.get_joint_angles()
    data[i]['qdot2'] = fetch.get_joint_vel()
    im = torch.tensor(fetch.get_image(True))
    data[i]['r3m2'] = r3m(im.permute(2,0,1).reshape(-1, 3, 224, 224)).cpu().numpy()
    data[i]['g2'] = fetch.gripper_open

def step_sim(i, fps, fetch, data, r3m):
    collect_before_data(i, fetch, r3m, data)
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
                                ('qdot1', np.float32, 7),
                                ('g1', np.float32, 3),
                                ('x1', np.float32, 3),
                                ('x2', np.float32, 3),
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
        for _ in range(10):
            dist = 1
            k = 0
            x = np.random.uniform(0.6, 0.8)
            y = np.random.uniform(-0.4, 0.4)
            z = np.random.uniform(0.3, 0.6)
            pos = np.array([x, y, z])
            while dist > 0.1 and k < 40:
                dist = fetch.move_to(pos)
                i = step_sim(i, fps, fetch, data, r3m)
                # print(dist, fetch.get_gripper_state())
                k += 1

        fetch.disconnect()


    data = data[:i]
    print(i)
    np.savez_compressed('random_data{}'.format(j), data=data)
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
        for i in range(30,40,3):
            processes = []
            for j in range(3):
                pr = mp.Process(target=pick_place, args=(i+j, r3m))
                pr.start()
                processes.append(pr)
            for pr in processes:
                pr.join()
    

