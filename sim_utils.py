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
        # p.setGravity(0,0,-9.81)
        planeId = p.loadURDF("plane.urdf")
        # tableId = p.loadURDF("table/table.urdf", (1, 0, 0), p.getQuaternionFromEuler((0, 0, 3.1415/2.0)))
        self.mugIds = []

        self.fetch = p.loadURDF("fetch_description/robots/fetch_obj.urdf")#, start_pos, start_orientation)
        self.arm_joints = [x for x in range(10,17)]

        #lock down fetch base and torso lift
        torso_pos = p.getLinkState(self.fetch, 4)[0]
        p.createConstraint(self.fetch, -1, -1, -1, p.JOINT_FIXED, [0,0,0],[0,0,0],[0,0,0])
        p.createConstraint(self.fetch, 4, -1, -1, p.JOINT_FIXED, [0,0,0], [0,0,0], torso_pos)

        #set joint limits
        numJoints = p.getNumJoints(self.fetch)
        self.lower_limits = np.zeros(16)
        self.upper_limits = np.zeros(16)
        self.range_limits = np.zeros(16)
        self.rest_poses = np.zeros(16)
        j = 0
        for i in range(numJoints):
            jointInfo = p.getJointInfo(self.fetch, i)
            qIndex = jointInfo[3]
            if qIndex > -1:
                if j==2:
                    # tpos = p.getJointState(self.fetch, i)[0]
                    self.lower_limits[j] = 0
                    self.upper_limits[j] = 0
                    self.rest_poses[j] = 0
                else:
                    self.lower_limits[j] = jointInfo[8]
                    self.upper_limits[j] = jointInfo[9]
                    if self.upper_limits[j] == -1:
                        self.upper_limits[j] = 6.28
                    self.rest_poses[j] = p.getJointState(self.fetch, i)[0]
                self.range_limits[j] = self.upper_limits[j] - self.lower_limits[j]
                j += 1

        print('low limits', self.lower_limits)
        print('high limits', self.upper_limits)
        print('ranges', self.range_limits)
        print('rest', self.rest_poses)
        self.rest_poses[5] = -1.57
        self.rest_poses[6] = 1.57
        self.rest_poses[3] = 5

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

    def generate_mugs(self, random_number=True):
        mug_x_lim = [0.7, 1.0] #limits for mug position
        mug_y_lim = [-.5, .5]
        if random_number:
            num_mugs = np.random.randint(1,5)
        else:
            num_mugs = 1
        for _ in range(num_mugs):
            mug_x = np.random.uniform(mug_x_lim[0], mug_x_lim[1], 1)
            mug_y = np.random.uniform(mug_y_lim[0], mug_y_lim[1], 1)
            self.mugIds.append(p.loadURDF("objects/mug.urdf", (mug_x, mug_y, 0.6)))
            p.changeDynamics(self.mugIds[-1], -1, lateralFriction=0.5) #reduce friction for a bit more realistic sliding

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
        # qdot is shape (7,)
        forces = np.ones(len(self.arm_joints)) * 1000
        p.setJointMotorControlArray(self.fetch, self.arm_joints, p.VELOCITY_CONTROL, targetVelocities = qdot, forces=forces)

    def get_joint_angles(self):
        states = p.getJointStates(self.fetch, self.arm_joints)
        q = [state[0] for state in states]
        return q
    
    def push_mug(self, dir=None): #will add direction

        #get mug position
        # mug_pos = p.getBasePositionAndOrientation(self.mugIds[0], 0)[0]
        # goal_pos = np.array(mug_pos)+[-0.0, 0, 0]
        goal_pos = p.getLinkState(self.fetch, 17)[0]
        goal_pos = np.array(goal_pos) + [-0.3, 0.1, 0.1]
        goal_config = p.calculateInverseKinematics(self.fetch, 17, goal_pos,
                                                   targetOrientation=[1,0,0,0],
                                                   lowerLimits=self.lower_limits,
                                                   upperLimits=self.upper_limits,
                                                   jointRanges=self.range_limits,
                                                   restPoses=self.rest_poses)
        # goal_config = p.calculateInverseKinematics(self.fetch, 17, goal_pos)
        
        numJoints = p.getNumJoints(self.fetch)
        j = 0
        for i in range(numJoints):
            jointInfo = p.getJointInfo(self.fetch, i)
            qIndex = jointInfo[3]
            if qIndex > -1:
                p.resetJointState(self.fetch,i,goal_config[qIndex-7])
                j += 1
        # p.setJointMotorControlArray(self.fetch, joints, p.POSITION_CONTROL, targetPositions=goal_config)
        # print('n joints', p.getNumJoints(self.fetch))
        print('goal pos', goal_pos)
        
        print('goal config', goal_config, 'len', len(goal_config))
