import pybullet as p
import pybullet_data
import time
import matplotlib.pyplot as plt
import numpy as np
import random
#from kinematics import joint, link
#from node import node


class belief_propagation():

    def __init__(self, graph):
        print("Starting belief prop...")
        # Take in a list of urdf files representing the links we are tracking

        # Connect to pybullet physics engine
        physicsClient = p.connect(p.GUI)

        # Upload each link
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0,0,-10)
        planeId = p.loadURDF("plane.urdf")
        tableId = p.loadURDF("table/table.urdf", (0.85, 0, 0), p.getQuaternionFromEuler((0, 0, 3.1415/2.0)))
        mugId = p.loadURDF("objects/mug.urdf", (0.55, 0,  0.6))
        #mugId = p.loadURDF("tray/tray_textured2.urdf", (0.5, 0.1,  0.6))
        self.fetch = p.loadURDF("fetch_description/robots/fetch_obj.urdf")#, start_pos, start_orientation)
        #time.sleep(10)
        for i in range(p.getNumJoints(self.fetch)):
            print(i)
            print(p.getJointState(self.fetch, i))
        time.sleep(2)
        print("STARTING TRY")
        p.setJointMotorControl2(self.fetch, 15, p.POSITION_CONTROL, targetPosition=3.14/2.0)
        print("DONE TRYING")
        

        # Set camera properties and positions
        self.img_width = 640
        self.img_height = 480

        self.cam_fov = 54
        self.img_aspect = self.img_width / self.img_height
        self.dpth_near = 0.02
        self.dpth_far = 5

        # Camera extrinsics calulated by (position of camera (xyz), position of target (xyz), and up vector for the camera)
        self.view_matrix = p.computeViewMatrix([0.15, 0, 1.05], [0.6, 0, 0.7], [0, 0, 1]) #NOTE: You can calculate the extrinsics with another function that takes position and euler angles
        
        # Camera intrinsics
        self.projection_matrix = p.computeProjectionMatrixFOV(self.cam_fov, self.img_aspect, self.dpth_near, self.dpth_far)

        pos = p.calculateInverseKinematics(self.fetch, 17, [0.55, 0, 0.6])
        print(pos)

        for i in range(10000):
            #print(p.getJointState(self.fetch, 6))
            self.get_image()
            print(p.getJointState(self.fetch, 15))
            p.stepSimulation()
            time.sleep(1./240.)

    # A single update given one image to use for computing the likelihood function
    def update(self, img):
        self.get_likelihoods()
        self.forward_pass()
        self.backward_pass()

    # Update likelihoods based on render and compare
    def get_image(self):
        # Get rgb, depth, and segmentation images
        images = p.getCameraImage(self.img_width,
                        self.img_height,
                        self.view_matrix,
                        self.projection_matrix,
                        shadow=True,
                        renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb_opengl = np.reshape(images[2], (self.img_height, self.img_width, 4)) * 1. / 255.
        depth_buffer_opengl = np.reshape(images[3], [self.img_height, self.img_width])
        depth_opengl = self.dpth_far * self.dpth_near / (self.dpth_far - (self.dpth_far - self.dpth_near) * depth_buffer_opengl)
        seg_opengl = np.reshape(images[4], [self.img_height, self.img_width]) #* 1. / 255.

        """plt.imshow(rgb_opengl) #, cmap='gray', vmin=0, vmax=1)
        plt.title('RGB OpenGL3')
        plt.show()
        plt.imshow(depth_opengl) #, cmap='gray', vmin=0, vmax=1)
        plt.title('Depth OpenGL3')
        plt.show()
        plt.imshow(seg_opengl) #, cmap='gray', vmin=0, vmax=1)
        plt.title('Seg OpenGL3')
        plt.show()"""

    # Send messages forward in the graph
    def forward_pass(self):
        for parent in self.forward_graph.keys():
            msgs = parent.get_msgs()
            for child in self.forward_graph[parent]:
                child.accept_msgs(msgs)

    # Send messages backward in the graph
    def backward_pass(self):
        for parent in self.backward_graph.keys():
            msgs = parent.get_msgs()
            for child in self.backward_graph[parent]:
                child.accept_msgs(msgs)

    def build_graph(self, forward_graph):
        # For each entry, take everything it connects to and have those connect to it in the backwards pass
        self.forward_graph = forward_graph
        for k, v in forward_graph.items():
            for n in v:
                if n not in self.backward_graph:
                    self.backward_graph[n] = []
                self.backward_graph[n].append(k)


if __name__ == '__main__':

    # Specify the graph for belief propagation
    connection_dict = {'shoulder_pan_link':['shoulder_lift_link'], 'shoulder_lift_link':['upperarm_roll_link'], 'upperarm_roll_link':['elbow_flex_link'],
                        'elbow_flex_link':['forearm_roll_link'], 'forearm_roll_link':['wrist_flex_link'], 'wrist_flex_link':['wrist_roll_link'], 'wrist_roll_link':['gripper_link']}

    bp = belief_propagation(connection_dict)
    #print("Forward Graph: ", bp.forward_graph)
    #print("Backward Graph: ", bp.backward_graph)
    #bp.get_likelihood()

    # Main update loop. Can update over video frames or repeat on one image
    """img = None
    for i in range(10):
        bp.update(img)"""

    time.sleep(10)
    p.disconnect()
