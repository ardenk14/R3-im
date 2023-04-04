import torch
from single_step_dataloader import FetchMotionDataset, get_dataloader
from behavior_cloning_net import BehaviorCloningModel
import matplotlib.pyplot as plt

# TODO: Take command arguments to give a file to save in or read from
if __name__ == '__main__':
    #dataset = FetchMotionDataset('data.npz')
    #print("Dataset: ", dataset)
    trainloader = get_dataloader('data/block_lift')
    print("train loader: ", trainloader)

    # Create model
    state_dim = 2048 + 8 # TODO: have dataloader function return these dimensions
    action_dim = 8
    model = BehaviorCloningModel(state_dim, action_dim)

    # Train forward model
    losses = model.train_model(trainloader, num_epochs=2000)

    # Save the model
    print("Saving...")
    torch.save(model.state_dict(), 'bc_model.pt')
    print("Saved at bc_model.pt")

    # Plot forward only losses
    plt.plot([i for i in range(len(losses))], losses)
    plt.show()