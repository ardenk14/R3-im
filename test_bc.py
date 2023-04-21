from bc_SingleBlockLift import test_bc
from r3m import load_r3m
import numpy as np
import torch
from models.single_step_dataloader import FetchMotionDataset, get_dataloader
from models.behavior_cloning_net import BehaviorCloningModel
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp

def eval_bc(attempts, r3m, row, col, q):
    n_demos = [i for i in range(5, 31, 5)]
    state_dim = 2048 + 7 + 3
    action_dim = 8
    if col == 1:
        fname = "models/bc_side2x2_{}.pt".format(n_demos[row])
        rand = 0.01
        ego = False
    elif col == 2:
        fname = "models/bc_side4x4_{}.pt".format(n_demos[row])
        rand = 0.02
        ego = False
    elif col == 3:
        fname = "models/bc_ego2x2_{}.pt".format(n_demos[row])
        rand = 0.01
        ego = True
    elif col == 4:
        fname = "models/bc_ego4x4_{}.pt".format(n_demos[row])
        rand = 0.02
        ego = True
    model = BehaviorCloningModel(state_dim, action_dim)
    model.load_state_dict(torch.load(fname))
    model.eval()
    success = test_bc(attempts, model, state_dim, r3m, rand, ego)
    q.put((row, col, success))

# TODO: Take command arguments to give a file to save in or read from
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    r3m = load_r3m("resnet50") # resnet18, resnet34
    r3m.eval()
    r3m.to(device)

    with torch.no_grad():
        scores = {'n_demonstrations':[], '2x2 side':[], '2x2 ego':[], '4x4 side':[], '4x4 ego':[]}
        scores = np.zeros((6, 5))
        scores[:, 0] = [i for i in range(5, 31, 5)]
        attempts = 30
        for idx, n_demonstrations in enumerate(range(5, 31, 5)):
            processes = []
            q = mp.Queue()
            for i in range(1,5):
                pr = mp.Process(target = eval_bc, args=(attempts, r3m, idx, i, q))
                pr.start()
                processes.append(pr)
            for pr in processes:
                pr.join()

            while not q.empty():
                item = q.get()
                print(item)
                scores[item[0], item[1]] = item[2]

    print(scores)
    data = pd.DataFrame(scores, columns=['n_demo', 'side 2x2', 'side 4x4', 'ego 2x2', 'ego 4x4'])
    data.to_csv('bc_scores2.csv', index=False, encoding='utf-8')

    ############ Training ###############
    # for idx, n_demonstrations in enumerate(range(5, 31, 5)):
    #     ########### side view 2x2 ###############
    #     trainloader = get_dataloader('data/eval/side2x2', batch_size=32, num_demonstrations=n_demonstrations)

    #     # Create model
    #     state_dim = 2048 + 7 + 3 # TODO: have dataloader function return these dimensions
    #     action_dim = 8
    #     model = BehaviorCloningModel(state_dim, action_dim)
    #     model.train()

    #     # Train forward model
    #     losses = model.train_model(trainloader, num_epochs=2000)
    #     model.eval()

    #     name = "bc_side2x2_{}.pt".format(n_demonstrations)
    #     torch.save(model.state_dict(), name)
    #     print("saved", name)
    #     # scores['2x2 side'].append(test_bc(10, model, state_dim, r3m, 0.01, False))

    #     ########### side view 4x4 ###############
    #     trainloader = get_dataloader('data/eval/side4x4', batch_size=32, num_demonstrations=n_demonstrations)

    #     # Create model
    #     state_dim = 2048 + 7 + 3 # TODO: have dataloader function return these dimensions
    #     action_dim = 8
    #     model = BehaviorCloningModel(state_dim, action_dim)
    #     model.train()

    #     # Train forward model
    #     losses = model.train_model(trainloader, num_epochs=2000)
    #     model.eval()

    #     name = "bc_side4x4_{}.pt".format(n_demonstrations)
    #     torch.save(model.state_dict(), name)
    #     print("saved", name)
    #     # scores['4x4 side'].append(test_bc(10, model, state_dim, r3m, 0.02, False))

    #     ########### ego 2x2 ###############
    #     trainloader = get_dataloader('data/eval/ego2x2', batch_size=32, num_demonstrations=n_demonstrations)

    #     # Create model
    #     state_dim = 2048 + 7 + 3 # TODO: have dataloader function return these dimensions
    #     action_dim = 8
    #     model = BehaviorCloningModel(state_dim, action_dim)
    #     model.train()

    #     # Train forward model
    #     losses = model.train_model(trainloader, num_epochs=2000)
    #     model.eval()

    #     name = "bc_ego2x2_{}.pt".format(n_demonstrations)
    #     torch.save(model.state_dict(), name)
    #     print("saved", name)
    #     # scores['2x2 ego'].append(test_bc(10, model, state_dim, r3m, 0.01, True))

    #     ########### ego 4x4 ###############
    #     trainloader = get_dataloader('data/eval/ego4x4', batch_size=32, num_demonstrations=n_demonstrations)

    #     # Create model
    #     state_dim = 2048 + 7 + 3 # TODO: have dataloader function return these dimensions
    #     action_dim = 8
    #     model = BehaviorCloningModel(state_dim, action_dim)
    #     model.train()

    #     # Train forward model
    #     losses = model.train_model(trainloader, num_epochs=2000)
    #     model.eval()


    #     name = "bc_ego4x4_{}.pt".format(n_demonstrations)
    #     torch.save(model.state_dict(), name)
    #     print("saved", name)
    #     # scores['4x4 ego'].append(test_bc(10, model, state_dim, r3m, 0.02, True))

    #     # print(scores)
    
