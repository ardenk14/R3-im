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

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

r3m = load_r3m("resnet50") # resnet18, resnet34
r3m.eval()
r3m.to(device)

if __name__ == '__main__':
    with torch.no_grad():
        fps = 15.0
        n_samples = 150 #samples (states) per run
        n_runs = 10 #number of runs, fetch is reinitialized each time

        #limits for velocity and acceleration
        max_v = 0.5
        max_a = 0.6
        max_j = 0.6

        #set up data type as [xt, qt, qdot, xt+1, ]
        step_dtype = np.dtype([('xt', np.float32, 2048),
                            ('qt', np.float32, 7),
                            ('qdott', np.float32, 7),
                            ('at', np.float32, 7),
                            ('at-1', np.float32, 7),
                            ('xt_1', np.uint8, 2048),
                            ('qt_1', np.float32, 7),
                            ('qdott_1', np.float32, 7),
                            ('xg', np.float32, 2048)])
        
        

        data = np.zeros((n_samples, n_runs), dtype=step_dtype)
        
        for run_idx in range(n_runs):
            fetch = SymFetch(gui=True)
            fetch.generate_mugs(random_color=True)
            
            for _ in range(200):
                p.stepSimulation()

            qdot = (2*np.random.rand(7) - 1)*max_v #get initial velocity in [-max_v, max_v]
            qddot = (2*np.random.rand(7) - 1)*max_a

            prev_a = qdot

            print('\n\n run', run_idx, '\n')
            for i in range(n_samples):

                qdddot = (2*np.random.rand(7) - 1)*max_j #jerk
                qddot += qdddot
                qddot = np.clip(qddot, -max_a, max_a)
                qdot += qddot
                qdot = np.clip(qdot, -max_v, max_v)

                fetch.set_arm_velocity(qdot)

                #collect images and state
                data[i,run_idx]['qt'] = fetch.get_joint_angles()
                im = torch.tensor(fetch.get_image(True))
                data[i,run_idx]['xt'] = r3m(im.permute(2,0,1).reshape(-1, 3, 224, 224)).cpu().numpy()
                data[i,run_idx]['at'] = qdot
                data[i,run_idx]['qdott'] = fetch.get_joint_vel()
                data[i,run_idx]['at-1'] = prev_a

                #advance sim
                for _ in range(int(240/fps)):
                    p.stepSimulation()
                    # time.sleep(1./240.)

                data[i,run_idx]['qt_1'] = fetch.get_joint_angles()
                im = torch.tensor(fetch.get_image(True))
                data[i,run_idx]['xt_1'] = r3m(im.permute(2,0,1).reshape(-1, 3, 224, 224)).cpu().numpy()
                data[i,run_idx]['qdott_1'] = fetch.get_joint_vel()
                data[i,run_idx]['xg'] = data[i,run_idx]['xt_1'] 
                prev_a = qdot
                
            time.sleep(1)
            p.disconnect()

        np.savez_compressed('data', data=data)
            