# R3-im

#### Goal: Zero-shot imitation of human actions. 

R3M has been shown to improve learning efficiency when applied to a type of imitation learning called behavior cloning, as it is able to pull key information for robot manipulation tasks from images. We aim to utilize R3M to allow robots to imitate human actions in a robust and data-efficient manner. A Goal-Conditioned Skill Policy (GSP) allows a robot to learn a policy that maps two states (images of the environment showing the current state and the goal state) to a sequence of actions between them. By encoding the images (state and goal/sub-goal) with a pre-trained R3M model (as opposed to an AlexNet), we expect to learn a more robust GSP with a smaller amount of training data.

GSP Diagram:
![GSP_diagram](https://user-images.githubusercontent.com/56897405/223612419-20bba4d1-fa39-4e00-b9d7-2e7bf0a34ac1.PNG)

References:
1) R3M: https://arxiv.org/pdf/2203.12601.pdf
2) GSP: https://pathak22.github.io/zeroshot-imitation/resources/iclr18.pdf
