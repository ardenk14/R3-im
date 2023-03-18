import pybullet as p
import pybullet_data
import time
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2

class SymFetch():

    def __init__(self, gui=True) -> None:
        # Connect to pybullet physics engine
        if gui:
            physicsClient = p.connect(p.GUI)
        else:
            physicsClient = p.connect(p.DIRECT)

        # Upload each link
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        p.setGravity(0,0,-9.81)
        planeId = p.loadURDF("plane.urdf")
        tableId = p.loadURDF("table/table.urdf", (1, 0, 0), p.getQuaternionFromEuler((0, 0, 3.1415/2.0)))
        self.mugIds = []

        self.fetch = p.loadURDF("fetch_description/robots/fetch_obj.urdf")#, start_pos, start_orientation)
        self.arm_joints = [x for x in range(10,17)]

        #lock down fetch base and torso lift
        torso_pos = p.getLinkState(self.fetch, 4)[0]
        p.createConstraint(self.fetch, -1, -1, -1, p.JOINT_FIXED, [0,0,0],[0,0,0],[0,0,0])
        p.createConstraint(self.fetch, 4, -1, -1, p.JOINT_FIXED, [0,0,0], [0,0,0], torso_pos)

        #set joint limits
        numJoints = p.getNumJoints(self.fetch)
        nLimits = 16
        self.lower_limits = []
        self.upper_limits = []
        self.range_limits = []
        self.rest_poses = []
        self.joint_damping = []
        j = 0
        for i in range(numJoints):
            jointInfo = p.getJointInfo(self.fetch, i)
            qIndex = jointInfo[3]
            # print(jointInfo)
            if qIndex > -1:
                if jointInfo[1] == b'torso_lift_joint':
                    # tpos = p.getJointState(self.fetch, i)[0]
                    self.lower_limits.append(0.0)
                    self.upper_limits.append(0.1)
                    self.rest_poses.append(0.0)
                    self.joint_damping.append(1000000)
                    # self.range_limits.append(0)
                else:
                    self.lower_limits.append(jointInfo[8])
                    self.upper_limits.append(jointInfo[9])
                    if self.upper_limits[-1] == -1:
                        self.upper_limits[-1] = 6.28
                    # self.lower_limits.append(-3.14)
                    self.rest_poses.append(p.getJointState(self.fetch, i)[0])
                    self.joint_damping.append(0)

                self.range_limits.append(self.upper_limits[-1] - self.lower_limits[-1])
                # self.range_limits.append(0)
                j += 1
        # print(j)
        # print('low limits', self.lower_limits)
        # print('high limits', self.upper_limits)
        # print('ranges', self.range_limits)
        # print('rest', self.rest_poses)

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

    def generate_mugs(self, random_number=True, random_color=False):
        mug_x_lim = [0.7, 1.0] #limits for mug position
        mug_y_lim = [-.5, .5]

        if random_number:
            num_mugs = np.random.randint(1,15)
        else:
            num_mugs = 1

        for _ in range(num_mugs):
            mug_x = np.random.uniform(mug_x_lim[0], mug_x_lim[1], 1)
            mug_y = np.random.uniform(mug_y_lim[0], mug_y_lim[1], 1)
            if random_color:
                urdf_file = np.random.choice(['./red_mug.urdf', './blue_mug.urdf', './dark_red_mug.urdf'])
            else:
                urdf_file = './red_mug.urdf'
            self.mugIds.append(p.loadURDF(urdf_file, (mug_x, mug_y, 0.6)))
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
    
    def get_joint_vel(self):
        states = p.getJointStates(self.fetch, self.arm_joints)
        q = [state[1] for state in states]
        return q
    
    def move_to_mug(self): #super janky rn

        #get mug position
        mug_pos = p.getBasePositionAndOrientation(self.mugIds[0], 0)[0]
        # goal_pos = np.array(mug_pos)+[-0.2, 0.0, 0.2]
        goal_pos = np.array(p.getLinkState(self.fetch, 16)[0])+ [-0.1, 0.0, 0]
        close_enough = False
        iters=0
        # while (not close_enough) and (iters < 10):
        goal_config = p.calculateInverseKinematics(self.fetch, 16, goal_pos,
                                                # p.getQuaternionFromEuler((0, 0, 0)),
                                                lowerLimits=self.lower_limits,
                                                upperLimits=self.upper_limits,
                                                jointRanges=self.range_limits,
                                                restPoses=self.rest_poses,
                                                maxNumIterations=50,
                                                jointDamping=self.joint_damping,
                                                solver=p.IK_DLS)
        # print('goal', goal_pos)
        # print(goal_config)
        
        numJoints = p.getNumJoints(self.fetch)
        j = 0
        for i in range(numJoints):
            jointInfo = p.getJointInfo(self.fetch, i)
            qIndex = jointInfo[3]
            if qIndex > -1:
                # p.resetJointState(self.fetch,i,goal_config[qIndex-7])
                # if j==2:
                #     p.setJointMotorControl2(self.fetch, i, p.POSITION_CONTROL, force=1000, targetPosition=0.1, maxVelocity=0.5)
                # else:
                p.setJointMotorControl2(self.fetch, i, p.POSITION_CONTROL, force=1000, targetPosition=goal_config[qIndex-7], maxVelocity=0.5)
                j += 1

            iters += 1
            diff = p.getLinkState(self.fetch, 17)[0] - goal_pos
            if np.linalg.norm(diff) < 0.1:
                close_enough = True

        

    def push_mug(self):
        #get mug position
        mug_pos = p.getBasePositionAndOrientation(self.mugIds[0], 0)[0]
        goal_pos = np.array(mug_pos)+[0.0, -0.5, 0.1]
        goal_config = p.calculateInverseKinematics(self.fetch, 17, goal_pos,
                                                   p.getQuaternionFromEuler((0, 0, 0)),
                                                   self.lower_limits,
                                                   self.upper_limits,
                                                   self.range_limits,
                                                   self.rest_poses)
        
        numJoints = p.getNumJoints(self.fetch)
        j = 0
        for i in range(numJoints):
            jointInfo = p.getJointInfo(self.fetch, i)
            qIndex = jointInfo[3]
            if qIndex > -1:
                # p.resetJointState(self.fetch,i,goal_config[qIndex-7])
                if j==2:
                    # p.resetJointState(self.fetch,i,0.3)
                    p.setJointMotorControl2(self.fetch, i, p.POSITION_CONTROL, targetPosition=0, maxVelocity=0.5)
                else:
                    p.setJointMotorControl2(self.fetch, i, p.POSITION_CONTROL, targetPosition=goal_config[qIndex-7], maxVelocity=0.5)
                j += 1
