import pybullet as p
import pybullet_data
import time
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
from sim_utils import SymFetch

if __name__ == '__main__':
    fps = 15.0
    n_samples = 150 #samples (states) per run
    n_runs = 10 #number of runs, fetch is reinitialized each time

    #limits for velocity and acceleration
    max_v = 0.5
    max_a = 0.6
    max_j = 0.6

    #set up data type as [xt, qt, qdot, xt+1, ]
    step_dtype = np.dtype([('xt', np.uint8, (224,224,3)),
                           ('qt', np.float32, 7),
                           ('qdot', np.float32, 7),
                           ('xt_1', np.uint8, (224,224,3)),
                           ('qt_1', np.float32, 7)])
    
    data = np.zeros((n_samples, n_runs), dtype=step_dtype)
    
    for run_idx in range(n_runs):
        fetch = SymFetch()
        fetch.generate_mugs()

        qdot = (2*np.random.rand(7) - 1)*max_v #get initial velocity in [-max_v, max_v]
        qddot = (2*np.random.rand(7) - 1)*max_a

        for i in range(n_samples):

            qdddot = (2*np.random.rand(7) - 1)*max_j #jerk
            qddot += qdddot
            qddot = np.clip(qddot, -max_a, max_a)
            qdot += qddot
            qdot = np.clip(qdot, -max_v, max_v)

            fetch.set_arm_velocity(qdot)

            #collect images and state
            data[i,run_idx]['qt'] = fetch.get_joint_angles()
            data[i,run_idx]['xt'] = fetch.get_image(True)
            data[i,run_idx]['qdot'] = qdot

            #advance sim
            for _ in range(int(240/fps)):
                p.stepSimulation()
                time.sleep(1./240.)

            data[i,run_idx]['qt_1'] = fetch.get_joint_angles()
            data[i,run_idx]['xt_1'] = fetch.get_image(True)
            
        time.sleep(1)
        p.disconnect()

    np.savez_compressed('data', data=data)
        