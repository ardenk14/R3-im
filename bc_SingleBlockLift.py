import pybullet as p
import pybullet_data
import time
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import torch
from sim_utils import SymFetch
from models.behavior_cloning_net import BehaviorCloningModel

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
        success = 0
        attempts = 10
        for j in range(attempts):
            fetch = SymFetch(random_init=False)
            fetch.generate_blocks(random_number=False, random_color=False, random_pos=False)

            state_dim = 2048 + 7 + 3 # 7 joints + 1 gripper + 2048 for R3M embedding
            action_dim = 7 + 1 # 7 joint position changes + gripper action 

            model = BehaviorCloningModel(state_dim, action_dim)
            model.load_state_dict(torch.load('bc_side_model.pt'))
            model.eval()

            bc_input = torch.zeros((state_dim), device=device)
            for i in range(3000):
                if i%24==0:
                    im = torch.tensor(fetch.get_image(True))

                    # Set inputs for policy: State (joint positions and velocities) and R3M embedding
                    features = r3m(im.permute(2,0,1).reshape(-1, 3, 224, 224))
                    bc_input[:2048] = features
                    # Get current state
                    current_state = torch.from_numpy(fetch.get_joint_angles())
                    bc_input[2048:-3] = current_state
                    # bc_input[-1] = float(fetch.gripper_open)
                    bc_input[-3:] = torch.tensor(fetch.get_gripper_state())

                    # Get output from policy
                    output = model(bc_input.view(1,-1)).detach().cpu().numpy()

                    # Set robot commands from policy
                    pos = output[0,:7] +  current_state.numpy()
                    # pos = output[:3] +  fetch.get_ee_pos()
                    open_gripper = bool(round(output[0,-1]))
                    fetch.set_joint_angles(pos)
                    # fetch.move_to(pos)
                    fetch.set_gripper(open=open_gripper)
                    print(i, open_gripper, bc_input[-3:])
                fetch.stepSimulation()

                block_pos = np.array(p.getBasePositionAndOrientation(fetch.blockIds[0], 0)[0])
                finger_pos = np.array(p.getLinkState(fetch.fetch, 18)[0])
                if block_pos[2] > 0.5 and np.linalg.norm(block_pos-finger_pos) < 0.1:
                    success += 1
                    print("-----success!------")
                    break
                time.sleep(1./240.)                
            fetch.disconnect()

        print("-----------------------------")
        print("Successful {}/{} {} rate".format(success, attempts, success/attempts))