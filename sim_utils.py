import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client
import time
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2

class SymFetch():

    def __init__(self, gui=True, random_init=True) -> None:
        # Connect to pybullet physics engine
        # if gui:
        #     physicsClient = p.connect(p.GUI)
        # else:
        #     physicsClient = p.connect(p.DIRECT)

        if gui:
            self._p = bullet_client.BulletClient(connection_mode=p.GUI)
        else:
            self._p = bullet_client.BulletClient()

        # Upload each link
        self._p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        print(pybullet_data.getDataPath())
        self._p.setGravity(0,0,-9.81)
        planeId = self._p.loadURDF("plane.urdf")
        tableId = self._p.loadURDF("table/table.urdf", (1, 0, -0.3), self._p.getQuaternionFromEuler((0, 0, 3.1415/2.0)))
        self.blockIds = []

        self.fetch = self._p.loadURDF("fetch_description/robots/fetch_obj.urdf", useFixedBase=1)#, start_pos, start_orientation)
        self.arm_joints = [x for x in range(10,17)]
        self.gripper_open = True

        #set joint limits
        numJoints = self._p.getNumJoints(self.fetch)
        self.lower_limits = []
        self.upper_limits = []
        self.range_limits = []
        self.rest_poses = []
        self.joint_damping = []
        j = 0
        for i in range(numJoints):
            jointInfo = self._p.getJointInfo(self.fetch, i)
            qIndex = jointInfo[3]
            # print(jointInfo)
            if qIndex > -1:
                self.lower_limits.append(jointInfo[8])
                self.upper_limits.append(jointInfo[9])
                if self.upper_limits[-1] == -1:
                    self.upper_limits[-1] = 6.28
                    self.lower_limits.append(-6.28)
                self.rest_poses.append(self._p.getJointState(self.fetch, i)[0])
                self.joint_damping.append(0)

                self.range_limits.append(self.upper_limits[-1] - self.lower_limits[-1])
                j += 1
        # print('low limits', self.lower_limits)
        # print('high limits', self.upper_limits)
        # print('ranges', self.range_limits)
        # print('rest', self.rest_poses)
        if random_init:
            for i, joint_idx in enumerate(self.arm_joints):
                self._p.resetJointState(self.fetch, joint_idx, np.random.uniform(self.lower_limits[i]*0.05, self.upper_limits[i]*0.05))
                # p.resetJointState(self.fetch, joint_idx, np.random.uniform(-0.7, 0.7))

        self._p.resetJointState(self.fetch, 12 ,1.3)
        self.set_gripper()
        self._p.enableJointForceTorqueSensor(self.fetch, 18, True)
        self._p.enableJointForceTorqueSensor(self.fetch, 19, True)


        # Set camera properties and positions
        self.img_width = 640
        self.img_height = 480

        self.cam_fov = 54
        self.img_aspect = self.img_width / self.img_height
        self.dpth_near = 0.02
        self.dpth_far = 5

        # Camera extrinsics calulated by (position of camera (xyz), position of target (xyz), and up vector for the camera)
        # self.view_matrix = p.computeViewMatrix([0.15, 0, 1.05], [0.6, 0, 0.7], [0, 0, 1]) #NOTE: You can calculate the extrinsics with another function that takes position and euler angles
        self.view_matrix = self._p.computeViewMatrix([1.2, 1.0, 0.8], [0.6, 0, 0.4], [0, 0, 1])

        # Camera intrinsics
        self.projection_matrix = self._p.computeProjectionMatrixFOV(self.cam_fov, self.img_aspect, self.dpth_near, self.dpth_far)

    def generate_blocks(self, random_number=True, random_color=False, random_pos=True):
        block_x_lim = [0.6, 0.8]#[0.7, 1.0] #limits for mug position
        block_y_lim = [-.4, .4]

        if random_number:
            num_mugs = np.random.randint(1,10)
        else:
            num_mugs = 1

        for _ in range(num_mugs):
            if random_pos:
                mug_x = np.random.uniform(block_x_lim[0], block_x_lim[1], 1)
                mug_y = np.random.uniform(block_y_lim[0], block_y_lim[1], 1)
            else:
                rand = 0.01
                mug_x = 0.65 + np.random.uniform(-rand, rand)
                mug_y = 0.3 + np.random.uniform(-rand, rand)

                #FOR TESTING GSP
                # mug_x = 0.7 + np.random.uniform(-rand, rand)
                # mug_y = 0.2 + np.random.uniform(-rand, rand)
            if random_color:
                urdf_file = np.random.choice(['./objects/red_block.urdf', './objects/blue_block.urdf', './objects/green_block.urdf'])
            else:
                urdf_file = './objects/red_block.urdf'
            self.blockIds.append(self._p.loadURDF(urdf_file, (mug_x, mug_y, 0.3)))
            # p.changeDynamics(self.mugIds[-1], -1, lateralFriction=0.5) #reduce friction for a bit more realistic sliding

    def get_image(self, resize=False):
        # Get rgb, depth, and segmentation images
        images = self._p.getCameraImage(self.img_width,
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
        self._p.setJointMotorControlArray(self.fetch, self.arm_joints, self._p.VELOCITY_CONTROL, targetVelocities = qdot, forces=forces)

    def set_joint_angles(self, q):
        self._p.setJointMotorControlArray(self.fetch, self.arm_joints, self._p.POSITION_CONTROL, targetPositions = q)

    def set_gripper(self, open=None):
        if open is not None:
            self.gripper_open = open
        if self.gripper_open:
            self._p.setJointMotorControl2(self.fetch, 18, self._p.POSITION_CONTROL, targetPosition=0.04, maxVelocity=0.1, force=50)
            self._p.setJointMotorControl2(self.fetch, 19, self._p.POSITION_CONTROL, targetPosition=0.04, maxVelocity=0.1, force=50)
        else:
            self._p.setJointMotorControl2(self.fetch, 18, self._p.POSITION_CONTROL, targetPosition=0.001, maxVelocity=0.1, force=30)
            self._p.setJointMotorControl2(self.fetch, 19, self._p.POSITION_CONTROL, targetPosition=0.001, maxVelocity=0.1, force=30)

    def get_joint_angles(self):
        states = self._p.getJointStates(self.fetch, self.arm_joints)
        q = np.array([state[0] for state in states])
        return q
    
    def get_joint_vel(self):
        states = self._p.getJointStates(self.fetch, self.arm_joints)
        q = np.array([state[1] for state in states])
        return q

    def get_ee_pos(self):
        return self._p.getLinkState(self.fetch, 17)[0]
    
    def get_gripper_state(self):
        pos1 = np.array(self._p.getLinkState(self.fetch, 18)[0])
        pos2 = np.array(self._p.getLinkState(self.fetch, 19)[0])
        dist = np.linalg.norm(pos1-pos2)
        return (dist, self._p.getJointState(self.fetch, 18)[2][1], self._p.getJointState(self.fetch, 19)[2][1])
    
    def move_to(self, goal_pos):
        goal_config = self._p.calculateInverseKinematics(self.fetch, 17, goal_pos,
                                                self._p.getQuaternionFromEuler((0, 1.57, 0)),
                                                lowerLimits=self.lower_limits,
                                                upperLimits=self.upper_limits,
                                                jointRanges=self.range_limits,
                                                restPoses=self.rest_poses,
                                                maxNumIterations=50,
                                                # jointDamping=self.joint_damping,
                                                solver=self._p.IK_DLS)
        numJoints = self._p.getNumJoints(self.fetch)
        j = 0
        for i in range(numJoints):
            jointInfo = self._p.getJointInfo(self.fetch, i)
            qIndex = jointInfo[3]
            if qIndex > -1 and i!=18 and i!=19:
                self._p.setJointMotorControl2(self.fetch, i, self._p.POSITION_CONTROL, targetPosition=goal_config[qIndex-7], maxVelocity=0.5)
                j += 1

        return np.linalg.norm(goal_pos - self.get_ee_pos())
        
    def move_to_block(self, idx=0, move_above=False, jitter=None):
        #get mug position
        mug_pos = self._p.getBasePositionAndOrientation(self.blockIds[idx])[0]
        if mug_pos[1] > 0:
            goal_pos = np.array(mug_pos) #+ [0.1, 0.2, 0.0]
        else: 
            goal_pos = np.array(mug_pos) #+ [0.1, -0.2, 0.0]
        # goal_pos = np.array(p.getLinkState(self.fetch, 16)[0])+ [-0.1, 0.0, 0]
        if move_above:
            goal_pos[2] += 0.1
        else:
            goal_pos[2] += 0.0
        if jitter is not None:
            goal_pos += jitter
        # print('goal', goal_pos)
        self.move_to(goal_pos)

        return np.linalg.norm(goal_pos - self.get_ee_pos())
    
    def stepSimulation(self):
        self._p.stepSimulation()

    def disconnect(self):
        self._p.disconnect()
