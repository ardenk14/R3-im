import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from sim_utils import SymFetch
import torch
from r3m import load_r3m
import tqdm
import multiprocessing as mp


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
        fetch.generate_blocks(random_number=False, random_color=False, random_pos=False) #generate many blocks
        
        data = np.zeros(n_samples, dtype=step_dtype)
        last_action = np.zeros(7)
        for _ in range(10):
            block_idx = np.random.randint(0, len(fetch.blockIds))
            # move above the block
            # jitter = np.hstack((np.random.uniform(-0.04, 0.04, 2), np.random.uniform(-0.02, 0.02)))
            fetch.set_gripper(open=True)
            # for _ in range(40):
            dist = 1
            k = 0
            while dist > 0.1 and k < 40:
                dist = fetch.move_to_block(idx=block_idx, move_above=True)
                q = fetch.get_joint_angles()
                i = step_sim(i, fps, fetch, data, r3m, last_action)
                last_action = fetch.get_joint_angles() - q
                # print(dist, fetch.get_gripper_state())
                k += 1
            
            dist = 1
            k = 0
            while dist > 0.095 and k < 40:
                dist = fetch.move_to_block(idx=block_idx, move_above=True)
                q = fetch.get_joint_angles()
                i = step_sim(i, fps, fetch, data, r3m, last_action)
                last_action = fetch.get_joint_angles() - q
                k += 1
                # print(dist, fetch.get_gripper_state())

            dist = 1
            k = 0
            while dist > 0.1 and k < 40:
                dist = fetch.move_to_block(idx=block_idx)
                q = fetch.get_joint_angles()
                i = step_sim(i, fps, fetch, data, r3m, last_action)
                last_action = fetch.get_joint_angles() - q
                k += 1
                # print(dist, fetch.get_gripper_state())
            
            # close gripper
            fetch.set_gripper(open=False)
            for _ in range(7):
                q = fetch.get_joint_angles()
                i = step_sim(i, fps, fetch, data, r3m, last_action)
                last_action = fetch.get_joint_angles() - q
                # print(dist, fetch.get_gripper_state())

            pos = fetch.get_ee_pos()
            pos = np.array(pos) + [0.0, 0.0, 0.1]

            dist = 1
            k = 0
            while dist > 0.1 and k < 40:
                dist = fetch.move_to(pos)
                q = fetch.get_joint_angles()
                i = step_sim(i, fps, fetch, data, r3m, last_action)
                last_action = fetch.get_joint_angles() - q
                # print(dist, fetch.get_gripper_state())
                k += 1

            block_x_lim = [0.55,0.75]
            block_y_lim = [-.4, .4]
            place_x = np.random.uniform(block_x_lim[0], block_x_lim[1])
            place_y = np.random.uniform(block_y_lim[0], block_y_lim[1])
            pos = np.array([place_x, place_y, 0.5])
            #move above place
            dist = 1
            k = 0
            while dist > 0.1 and k < 40:
                dist = fetch.move_to(pos)
                q = fetch.get_joint_angles()
                i = step_sim(i, fps, fetch, data, r3m, last_action)
                last_action = fetch.get_joint_angles() - q
                k += 1
                # print(dist, fetch.get_gripper_state())

            pos = np.array([place_x, place_y, 0.4])
            #lower
            dist = 1
            k = 0
            while dist > 0.1 and k < 40:
                dist = fetch.move_to(pos)
                q = fetch.get_joint_angles()
                i = step_sim(i, fps, fetch, data, r3m, last_action)
                last_action = fetch.get_joint_angles() - q
                k += 1
                # print(dist, fetch.get_gripper_state())

            # open gripper
            fetch.set_gripper(open=True)
            for _ in range(7):
                q = fetch.get_joint_angles()
                i = step_sim(i, fps, fetch, data, r3m, last_action)
                last_action = fetch.get_joint_angles() - q
                # print(dist, fetch.get_gripper_state())

            pos = fetch.get_ee_pos()
            pos = np.array(pos) + [0.0, 0.0, 0.1]

            dist = 1
            k = 0
            while dist > 0.05 and k < 40:
                dist = fetch.move_to(pos)
                q = fetch.get_joint_angles()
                i = step_sim(i, fps, fetch, data, r3m, last_action)
                last_action = fetch.get_joint_angles() - q
                # print(dist, fetch.get_gripper_state())
                k += 1

        fetch.disconnect()


    data = data[:i]
    print(i)
    np.savez_compressed('pick_place_data{}'.format(j), data=data)
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
        for i in range(72,85,2):
            processes = []
            for j in range(2):
                pr = mp.Process(target=pick_place, args=(i+j, r3m))
                pr.start()
                processes.append(pr)
            for pr in processes:
                pr.join()
    

