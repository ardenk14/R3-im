import pybullet as p
import pybullet_data
import time
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2

class SymFetch():

    def __init__(self) -> None:
        # Connect to pybullet physics engine
        physicsClient = p.connect(p.GUI)

        # Upload each link
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0,0,-9.81)
        planeId = p.loadURDF("plane.urdf")
        tableId = p.loadURDF("table/table.urdf", (1, 0, 0), p.getQuaternionFromEuler((0, 0, 3.1415/2.0)))

        mug_x_lim = [0.7, 1.0] #limits for mug position
        mug_y_lim = [-.5, .5]
        num_mugs = np.random.randint(1,5)
        mugIds = []
        for _ in range(num_mugs):
            mug_x = np.random.uniform(mug_x_lim[0], mug_x_lim[1], 1)
            mug_y = np.random.uniform(mug_y_lim[0], mug_y_lim[1], 1)
            mugIds.append(p.loadURDF("objects/mug.urdf", (mug_x, mug_y, 0.6)))
            p.changeDynamics(mugIds[-1], -1, lateralFriction=0.5) #reduce friction for a bit more realistic sliding

        self.fetch = p.loadURDF("fetch_description/robots/fetch_obj.urdf")#, start_pos, start_orientation)
        self.arm_joints = [x for x in range(10,17)]

        #lock down fetch base and torso lift
        torso_pos = p.getLinkState(self.fetch, 4)[0]
        p.createConstraint(self.fetch, -1, -1, -1, p.JOINT_FIXED, [0,0,0],[0,0,0],[0,0,0])
        p.createConstraint(self.fetch, 4, -1, -1, p.JOINT_FIXED, [0,0,0], [0,0,0], torso_pos)

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


    def get_image(self, resize=False):
        # Get rgb, depth, and segmentation images
        images = p.getCameraImage(self.img_width,
                        self.img_height,
                        self.view_matrix,
                        self.projection_matrix,
                        shadow=True,
                        renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb_img = np.reshape(images[2], (self.img_height, self.img_width, 4))[:,:,:3]
        
        #resize to 224,224
        if resize:
            rgb_img = cv2.resize(rgb_img, (224,224))
        
        return rgb_img
    
    def set_arm_velocity(self, qdot):
        # qdot is shape 7 elements
        p.setJointMotorControlArray(self.fetch, self.arm_joints, p.VELOCITY_CONTROL, targetVelocities = qdot)

    def get_joint_angles(self):
        states = p.getJointStates(self.fetch, self.arm_joints)
        q = [state[0] for state in states]
        return q
    



if __name__ == '__main__':
    fps = 15.0
    n_samples = 150 #samples (states) per run
    n_runs = 10 #number of runs, fetch is reinitialized each time

    #limits for velocity and acceleration
    max_v = 0.5
    max_a = 0.6
    max_j = 0.6

    #set up data type as [xt, qt, qdot, xt+1, ]
    step_dtype = np.dtype([('xt', np.uint8, (224,224,3)),
                           ('qt', np.float32, 7),
                           ('qdot', np.float32, 7),
                           ('xt_1', np.uint8, (224,224,3)),
                           ('qt_1', np.float32, 7)])
    
    data = np.zeros((n_samples,n_runs), dtype=step_dtype)
    
    for run_idx in range(n_runs):
        fetch = SymFetch()

        qdot = (2*np.random.rand(7) - 1)*max_v #get initial velocity in [-max_v, max_v]
        qddot = (2*np.random.rand(7) - 1)*max_a

        for i in range(n_samples):

            qdddot = (2*np.random.rand(7) - 1)*max_j #jerk
            qddot += qdddot
            qddot = np.clip(qddot, -max_a, max_a)
            qdot += qddot
            qdot = np.clip(qdot, -max_v, max_v)

            fetch.set_arm_velocity(qdot)

            #collect images and state
            data[i,run_idx]['qt'] = fetch.get_joint_angles()
            data[i,run_idx]['xt'] = fetch.get_image(True)
            data[i,run_idx]['qdot'] = qdot

            #advance sim
            for _ in range(int(240/fps)):
                p.stepSimulation()
                time.sleep(1./240.)

            data[i,run_idx]['qt_1'] = fetch.get_joint_angles()
            data[i,run_idx]['xt_1'] = fetch.get_image(True)
            
        time.sleep(1)
        p.disconnect()
    np.savez_compressed('data', data=data)
        