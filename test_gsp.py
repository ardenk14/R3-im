import pybullet as p
import pybullet_data
import time
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import torch
from sim_utils import SymFetch
from models.gsp_net import GSPNet
from models.goal_recognizer_net import GoalReconizerNet

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
        fetch = SymFetch(random_init=False)
        fetch.generate_blocks(random_number=False, random_color=False, random_pos=False)

        state_dim = 2048
        joint_state_dim = 7 + 3 # 7 joints + 1 gripper + 2048 for R3M embedding
        action_dim = 7 + 1 # 7 joint position changes + gripper action 

        model = GSPNet(state_dim, joint_state_dim, action_dim, num_actions=1)
        model.load_state_dict(torch.load('GSP_model.pt'))
        model.eval()

        gr = GoalReconizerNet(state_dim)
        gr.load_state_dict(torch.load('GoalRecognizer_net.pt'))
        gr.eval()

        goals = torch.from_numpy(np.load('goal.npy')).to(device).float()
        last_action = torch.zeros(1, action_dim).to(device)
        gsp_input = torch.zeros((state_dim), device=device)
        i = 0
        gr_score = 0

        goal_idx = 0
        goal = goals[goal_idx].view(1,-1)
        goal_idx += 1


        # dist = 1
        # k = 0
        # while dist > 0.095 and k < 40:
        #     dist = fetch.move_to_block(move_above=True)
        #     for _ in range(24):
        #         fetch.stepSimulation()
        #         time.sleep(1/240)
        #     k += 1
        
        try:
            while goal_idx < len(goals):
                while gr_score < 0.9:
                    if i%24==0:
                        im = torch.tensor(fetch.get_image(True))

                        # Set inputs for policy: State (joint positions and velocities) and R3M embedding
                        features = r3m(im.permute(2,0,1).reshape(-1, 3, 224, 224))
                        # Get current state
                        joint_state = torch.from_numpy(fetch.get_joint_angles())
                        joint_state = torch.cat((joint_state, torch.tensor(fetch.get_gripper_state()))).to(device).view(1,-1).float()

                        # Get output from policy
                        output = model(features, joint_state, goal, last_action)
                        last_action = output
                        output = output.detach().cpu().numpy()
                        # Set robot commands from policy
                        pos = output[0,:7] +  fetch.get_joint_angles()
                        # pos = output[:3] +  fetch.get_ee_pos()
                        open_gripper = bool(round(output[0,-1]))
                        fetch.set_joint_angles(pos)
                        # fetch.move_to(pos)
                        fetch.set_gripper(open=open_gripper)
                        gr_score = gr(torch.cat((features, goal), dim=-1))

                        dist = ((features - goal)**2).sum().sqrt()
                        print(i, gr_score.item(), dist.item())
                    fetch.stepSimulation()
                    i+=1
                    time.sleep(1./240.)

                print('----goal {}-------'.format(goal_idx))
                goal = goals[goal_idx].view(1,-1)
                goal_idx += 1
                gr_score = 0
                time.sleep(5)
        except KeyboardInterrupt as e:
            p.disconnect()