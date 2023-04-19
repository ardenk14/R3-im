import torch
from goal_recognizer_dataloader import get_dataloader
from goal_recognizer_net import GoalReconizerNet
import matplotlib.pyplot as plt

if __name__ == '__main__':
    #dataset = FetchMotionDataset('data.npz')
    #print("Dataset: ", dataset)
    trainloader = get_dataloader('./data/gsp2', batch_size=128)
    # print("train loader: ", trainloader)

    # Create model
    state_dim = 2048 # TODO: have dataloader function return these dimensions
    joint_state_dim = 0
    model = GoalReconizerNet(state_dim, joint_state_dim)
    model.train()
    # Train forward model
    losses = model.train_goal_recognizer(trainloader, num_epochs=500)

    # Save the model
    print("Saving...")
    torch.save(model.state_dict(), 'GoalRecognizer_net.pt')
    print("Saved at GoalRecognizer_net.pt")

    # Plot forward only losses
    plt.plot([i for i in range(len(losses))], losses)
    plt.show()