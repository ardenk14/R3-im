import torch
from single_step_dataloader import FetchMotionDataset, get_dataloader
from gsp_net import GSPNet
import matplotlib.pyplot as plt

# TODO: Take command arguments to give a file to save in or read from
if __name__ == '__main__':
    #dataset = FetchMotionDataset('data.npz')
    #print("Dataset: ", dataset)
    trainloader = get_dataloader('./data/gsp2')
    print("train loader: ", trainloader)

    # Create model
    state_dim = 2048 # TODO: have dataloader function return these dimensions
    action_dim = 8
    joint_state_dim = 7 + 3
    model = GSPNet(state_dim, joint_state_dim, action_dim, num_actions=1)
    model.train()
    # Train forward model
    forward_only_losses = model.train_forward_only(trainloader, num_epochs=50)

    # Train full model
    full_model_losses = model.train_full_model(trainloader, num_epochs=150)

    # Save the model
    print("Saving...")
    torch.save(model.state_dict(), 'GSP_model_long_horizon.pt')
    print("Saved at GSP_model_long_horizon.pt")

    # Plot forward only losses
    plt.plot([i for i in range(len(forward_only_losses))], forward_only_losses)
    plt.show()

    # Plot full model losses
    plt.plot([i for i in range(len(full_model_losses))], full_model_losses)
    plt.show()
