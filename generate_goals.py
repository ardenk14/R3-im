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

goal_frequency = 4

if __name__ == '__main__':
    with torch.no_grad():
        # if torch.cuda.is_available():
        #     device = "cuda"
        # else:
        #     device = "cpu"

        # r3m = load_r3m("resnet50")
        # r3m.to(device)
        # r3m.eval()

        i = 0
        fetch = SymFetch(gui=True, random_init=False)
        fetch.generate_blocks(random_number=False, random_color=False, random_pos=False) #generate one block
        
        goal_dtype = np.dtype([('r3m', np.float32, (224,224,3)),
                                ('x', np.float32, 3)])
        data = np.zeros(50, dtype=goal_dtype)

        # im = torch.tensor(fetch.get_image(True))
        # data[i,:] = r3m(im.permute(2,0,1).reshape(-1, 3, 224, 224)).cpu().numpy()
        # i+=1
        # data[i,:] = fetch.get_image(True)
        # i+=1

        dist = 1
        k = 0
        pos = np.array([0.7, 0.0, 0.5])
        while dist > 0.095 and k < 40:
            dist = fetch.move_to(pos)
            for _ in range(24):
                fetch.stepSimulation()
                time.sleep(1/240)
            k += 1

        #move above
        dist = 1
        k = 0
        while dist > 0.095 and k < 40:
            dist = fetch.move_to_block(move_above=True)
            for _ in range(24):
                fetch.stepSimulation()
                time.sleep(1/240)
            k += 1
            print(k, dist, fetch.get_gripper_state())

            if k%goal_frequency==0:
                data[i]['r3m'] = fetch.get_image(True)
                data[i]['x'] = fetch.get_ee_pos()
                i+=1
                time.sleep(1)

        data[i]['r3m'] = fetch.get_image(True)
        data[i]['x'] = fetch.get_ee_pos()
        i+=1

        #lower gripper
        dist = 1
        k = 0
        while dist > 0.1 and k < 40:
            dist = fetch.move_to_block(move_above=False)
            for _ in range(24):
                fetch.stepSimulation()
                time.sleep(1/240)
            k += 1
            print(dist, fetch.get_gripper_state())

            if k%goal_frequency==0:
                data[i]['r3m'] = fetch.get_image(True)
                data[i]['x'] = fetch.get_ee_pos()
                i+=1
                time.sleep(1)
        data[i]['r3m'] = fetch.get_image(True)
        data[i]['x'] = fetch.get_ee_pos()
        i+=1
        
        # close gripper
        fetch.set_gripper(open=False)
        for _ in range(50):
            fetch.stepSimulation()

        data[i]['r3m'] = fetch.get_image(True)
        data[i]['x'] = fetch.get_ee_pos()
        i+=1


        #lift block
        pos = fetch.get_ee_pos()
        pos = np.array(pos) + [0.0, 0.0, 0.2]
        dist = 1
        k = 0
        while dist > 0.05 and k < 40:
            dist = fetch.move_to(pos)
            for _ in range(24):
                fetch.stepSimulation()
                time.sleep(1/240)
            k += 1
            print(dist, fetch.get_gripper_state())

            if k%goal_frequency==0:
                data[i]['r3m'] = fetch.get_image(True)
                data[i]['x'] = fetch.get_ee_pos()
                i+=1
                time.sleep(1)
        data[i]['r3m'] = fetch.get_image(True)
        data[i]['x'] = fetch.get_ee_pos()
        i+=1


        print('number of goals: ', i)
        np.save('goal', data[:i])


    time.sleep(4)