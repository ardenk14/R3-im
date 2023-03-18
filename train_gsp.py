import torch
from load_data import FetchMotionDataset, get_dataloader
from gsp_net import GSPNet
import matplotlib.pyplot as plt

# TODO: Take command arguments to give a file to save in or read from
if __name__ == '__main__':
    #dataset = FetchMotionDataset('data.npz')
    #print("Dataset: ", dataset)
    trainloader = get_dataloader('data.npz')
    print("train loader: ", trainloader)

    # Create model
    state_dim = 2048 # TODO: have dataloader function return these dimensions
    action_dim = 7
    model = GSPNet(state_dim, action_dim)

    # Train forward model
    forward_only_losses = model.train_forward_only(trainloader, num_epochs=10)

    # Train full model
    full_model_losses = model.train_full_model(trainloader, num_epochs=20)

    # Save the model
    print("Saving...")
    torch.save(model.state_dict(), 'GSP_model.pt')
    print("Saved at GSP_model.pt")

    # Plot forward only losses
    plt.plot([i for i in range(len(forward_only_losses))], forward_only_losses)
    plt.show()

    # Plot full model losses
    plt.plot([i for i in range(len(full_model_losses))], full_model_losses)
    plt.show()
