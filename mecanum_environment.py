### To create a pybuller environment for the mecanum robot

import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data
import pybullet_planning as pp
import numpy as np
import time
import re
import os


class MechRoboEnv:
    def __init__(self, gui = True, timestep = 1/480, base_position = (0,0,0), benchmarking = False):

        self.n = 50
        self.c = 7

        self.points_list = []
        
        self.client_id = bc.BulletClient(p.GUI if gui else p.DIRECT)            # Initialize the bullet client
        self.client_id.setAdditionalSearchPath(pybullet_data.getDataPath())     # Add pybullet's data package to path
        self.client_id.setTimeStep(timestep)                                    # Set simulation timestep
        self.client_id.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)        # Disable Shadows
        self.client_id.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)            # Disable Frame Axes

        self.timestep = timestep
        self.benchmarking = benchmarking
        self.num_collisions = 0
        p.setAdditionalSearchPath(pybullet_data.getDataPath())      # Add pybullet's data package to path   

        self.colors = {'black': np.array([0., 0., 0.]) / 255.0,
                       'blue': np.array([20, 30, 255]) / 255.0,  # blue
                       'green': np.array([0, 255, 0]) / 255.0,  # green
                       'dark_blue': np.array([78, 121, 167]) / 255.0,
                       'dark_green': np.array([89, 161, 79]) / 255.0,
                       'brown': np.array([156, 117, 95]) / 255.0,  # brown
                       'orange': np.array([242, 142, 43]) / 255.0,  # orange
                       'yellow': np.array([237, 201, 72]) / 255.0,  # yellow
                       'gray': np.array([186, 176, 172]) / 255.0,  # gray
                       'red': np.array([255, 0, 0]) / 255.0,  # red
                       'purple': np.array([176, 122, 161]) / 255.0,  # purple
                       'cyan': np.array([118, 183, 178]) / 255.0,  # cyan
                       'light_cyan': np.array([0, 255, 255]) / 255.0,
                       'light_pink': np.array([255, 105, 180]) / 255.0,
                       'pink': np.array([255, 157, 167]) / 255.0}  # pink
                       
        
        target = self.client_id.getDebugVisualizerCamera()[11]          # Get cartesian coordinates of the camera's focus
        self.client_id.resetDebugVisualizerCamera(                      # Reset initial camera position
            cameraDistance=1.5,
            cameraYaw=90,
            cameraPitch=-25,
            cameraTargetPosition=target,
        )
        
        p.resetSimulation()                             
        self.client_id.setGravity(0, 0, -9.8)           # Set Gravity

        self.plane = self.client_id.loadURDF("plane.urdf", basePosition=(0, 0, 0), useFixedBase=True)   # Load a floor
        
        self.client_id.changeDynamics(                  # Set physical properties of the floor
            self.plane,
            -1,
            lateralFriction=1.1,
            restitution=0.5,
            linearDamping=0.5,
            angularDamping=0.5,
        )

        self.base_position = base_position

        self.obs_ids = []
    
    def spawn_cuboids(self, cuboid_config, color):

        for i in range(cuboid_config.shape[0]):
            
            vuid = self.client_id.createVisualShape(p.GEOM_BOX, 
                                        halfExtents = cuboid_config[i, 7:]/2,
                                        rgbaColor = np.hstack([self.colors[color], np.array([1.0])]))

            obs_id = self.client_id.createMultiBody(baseVisualShapeIndex = vuid, 
                                                basePosition = cuboid_config[i, :3], 
                                                baseOrientation = cuboid_config[i, 3:7])
            
            self.obs_ids.append(obs_id)

    def spawn_cylinders(self, cylinder_config, color):

        for i in range(cylinder_config.shape[0]):
            
            vuid = self.client_id.createVisualShape(p.GEOM_CYLINDER, 
                                        radius = cylinder_config[i, 7],
                                        length = cylinder_config[i, 8],
                                        rgbaColor = np.hstack([self.colors[color], np.array([1.0])]))

            obs_id = self.client_id.createMultiBody(baseVisualShapeIndex = vuid, 
                                                basePosition = cylinder_config[i, :3], 
                                                baseOrientation = cylinder_config[i, 3:7])
            
            self.obs_ids.append(obs_id)

    def spawn_spheres(self, sphere_config, color):

        for i in range(sphere_config.shape[0]):
            
            vuid = self.client_id.createVisualShape(p.GEOM_SPHERE, 
                                        radius = sphere_config[i, 7],
                                        rgbaColor = np.hstack([self.colors[color], np.array([1.0])]))

            obs_id = self.client_id.createMultiBody(baseVisualShapeIndex = vuid, 
                                                basePosition = sphere_config[i, :3], 
                                                baseOrientation = sphere_config[i, 3:7])
            
            self.obs_ids.append(obs_id)
    
    def spawn_collision_cuboids(self, cuboid_config, color = None, grasp = False):

        if grasp == True:
            baseMass = 0.001
            # color = random.choice(["blue","red","green","black"])
            color = 'black'
        else:
            baseMass = 0.
            if not color:
                color = 'yellow'

        for i in range(cuboid_config.shape[0]):
            
            cuid = self.client_id.createCollisionShape(p.GEOM_BOX, 
                                                       halfExtents = cuboid_config[i, 7:]/2)
            
            vuid = self.client_id.createVisualShape(p.GEOM_BOX, 
                                                    halfExtents = cuboid_config[i, 7:]/2,
                                                    rgbaColor = np.hstack([self.colors[color], np.array([1.0])]))

            obs_id = self.client_id.createMultiBody(baseMass = baseMass,
                                                    baseCollisionShapeIndex = cuid,
                                                    baseVisualShapeIndex = vuid, 
                                                    basePosition = cuboid_config[i, :3], 
                                                    baseOrientation = cuboid_config[i, 3:7])
            
            self.obs_ids.append(obs_id)

    def spawn_collision_cylinders(self, cylinder_config, color = None, grasp = False):

        if grasp == True:
            baseMass = 0.001
            color = 'green'
        else:
            baseMass = 0.
            color = 'yellow'

        for i in range(cylinder_config.shape[0]):
            
            cuid = self.client_id.createCollisionShape(p.GEOM_CYLINDER, 
                                                       radius = cylinder_config[i, 7],
                                                       height = cylinder_config[i, 8])                                                   
            
            vuid = self.client_id.createVisualShape(p.GEOM_CYLINDER, 
                                                    radius = cylinder_config[i, 7],
                                                    length = cylinder_config[i, 8],
                                                    rgbaColor = np.hstack([self.colors[color], np.array([1.0])]))

            obs_id = self.client_id.createMultiBody(baseMass = baseMass,
                                                    baseCollisionShapeIndex = cuid, 
                                                    baseVisualShapeIndex = vuid, 
                                                    basePosition = cylinder_config[i, :3], 
                                                    baseOrientation = cylinder_config[i, 3:7])
            
            self.obs_ids.append(obs_id)

    def spawn_collision_spheres(self, sphere_config, color = None, grasp = False):

        if grasp == True:
            baseMass = 0.001
            color = 'red'
        else:
            baseMass = 0.
            color = 'yellow'
        
        for i in range(sphere_config.shape[0]):
            
            cuid = self.client_id.createCollisionShape(p.GEOM_SPHERE, 
                                                       radius = sphere_config[i, 7])                                                   
            
            vuid = self.client_id.createVisualShape(p.GEOM_SPHERE, 
                                                    radius = sphere_config[i, 7],
                                                    rgbaColor = np.hstack([self.colors[color], np.array([1.0])]))

            obs_id = self.client_id.createMultiBody(baseMass = baseMass,
                                                    baseCollisionShapeIndex = cuid, 
                                                    baseVisualShapeIndex = vuid, 
                                                    basePosition = sphere_config[i, :3], 
                                                    baseOrientation = sphere_config[i, 3:7])
            
            self.obs_ids.append(obs_id)
    
    def clear_obstacles(self):

        for id in self.obs_ids:
            self.client_id.removeBody(id)
        self.obs_ids = []
    
    def clear_bounding_boxes(self):

        for obj_id in self.link_bounding_objs:
            self.client_id.removeBody(obj_id)
    
    def draw_frame(self, transform, scale_factor = 0.2):

        unit_axes_world = np.array([[scale_factor, 0, 0], 
                                    [0, scale_factor, 0], 
                                    [0, 0, scale_factor],
                                    [1, 1, 1]])
        axis_points = ((transform @ unit_axes_world)[:3, :]).T
        axis_center = transform[:3, 3]

        l1 = self.client_id.addUserDebugLine(axis_center, axis_points[0], self.colors['red'], lineWidth = 4)
        l2 = self.client_id.addUserDebugLine(axis_center, axis_points[1], self.colors['green'], lineWidth = 4)
        l3 = self.client_id.addUserDebugLine(axis_center, axis_points[2], self.colors['blue'], lineWidth = 4)

        frame_id = [l1, l2, l3]

        return frame_id[:]
    
    def remove_frame(self, frame_id):

        for id in frame_id:
            self.client_id.removeUserDebugItem(id)
        
    def inverse_of_transform(self, matrix):

        # Extract the rotation part and translation part of the matrix
        rotation_part = matrix[:3, :3]
        translation_part = matrix[:3, 3]
        
        # Calculate the inverse of the rotation part
        inverse_rotation = np.linalg.inv(rotation_part)
        
        # Calculate the new translation by applying the inverse rotation
        inverse_translation = -inverse_rotation.dot(translation_part)
        
        # Create the inverse transformation matrix
        inverse_matrix = np.zeros_like(matrix)
        inverse_matrix[:3, :3] = inverse_rotation
        inverse_matrix[:3, 3] = inverse_translation
        inverse_matrix[3, 3] = 1.0
        
        return inverse_matrix.copy()    
    
    
    def get_jacobian(self, joint_angles):

        T_EE = self.get_ee_tf_mat(joint_angles)

        J = np.zeros((6, 10))
        T = np.identity(4)
        for i in range(7 + 3):
            T = T @ self.get_tf_mat(i, joint_angles)

            p = T_EE[:3, 3] - T[:3, 3]
            z = T[:3, 2]

            J[:3, i] = np.cross(z, p)
            J[3:, i] = z

        return J[:, :7]
    
    def pose_to_transformation(self, pose):

        pos = pose[:3]
        quat = pose[3:]

        rotation_matrix = self.quaternion_to_rotation_matrix(quat)

        transform = np.zeros((4, 4))
        transform[:3, :3] = rotation_matrix.copy()
        transform[:3, 3] = pos.copy()
        transform[3, 3] = 1

        return transform

    def euler_to_rotation_matrix(yaw, pitch, roll):
        
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

        Rx = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

        R = Rz @ (Ry @ Rx)
        
        return R
    
    def quaternion_to_rotation_matrix(self, quat):
        """
        Convert a quaternion to a rotation matrix.
        
        :param q: Quaternion [w, x, y, z]
        :return: 3x3 rotation matrix
        """
        # w, x, y, z = quat
        # rotation_matrix = np.array([[1 - 2*y**2 - 2*z**2,  2*x*y - 2*z*w,        2*x*z + 2*y*w],
        #                             [2*x*y + 2*z*w,        1 - 2*x**2 - 2*z**2,  2*y*z - 2*x*w],
        #                             [2*x*z - 2*y*w,        2*y*z + 2*x*w,        1 - 2*x**2 - 2*y**2]])
        
        mat = np.array(self.client_id.getMatrixFromQuaternion(quat))
        rotation_matrix = np.reshape(mat, (3, 3))

        return rotation_matrix
        
    def rotation_matrix_to_quaternion(self, R):
        
        trace = np.trace(R)

        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S

        return np.array([qx, qy, qz, qw])
    
    def transformation_to_pose(self, T):

        pose = np.zeros((7,))
        pose[:3] = T[:3, 3]
        pose[3:] = self.rotation_matrix_to_quaternion(T[:3, :3])

        return pose.copy()
    
    def spawn_points(self, trajectory, color = 'black'):
        """
        Expected trajectory shape is (7, num_waypoints)
        """

        points = np.zeros((trajectory.shape[1] - 2, 3))
        for i in range(1, trajectory.shape[1] - 1):
            points[i-1] = self.forward_kinematics(trajectory[:, i])[:3, 3]
        colors = np.repeat([self.colors[color]], trajectory.shape[1] - 2, axis = 0)

        self.points_id = self.client_id.addUserDebugPoints(points, colors, 15)
        self.points_list.append(self.points_id)

    def remove_all_points(self):

        for id in self.points_list:
            self.client_id.removeUserDebugItem(id)
        self.points_list = []

    
    def draw_coordinate_system(self, position: np.array, rotation_matrix: np.array = np.eye(3), length: float = 0.1, width: float = 2.0, life_time: float = 0):
        """Draws a coordinate system at a given position using the specified orientation.
        
        Args:
            position (np.array): The 3D position of the coordinate system.
            orientation (np.array): The quaternion representing the orientation of the coordinate system.
            length (float): The length of the lines. Defaults to 0.1.
            width (float): The width of the lines. Defaults to 2.0.
            life_time (float): How long the coordinate system remains before despawning.
        """
        # Convert quaternion to rotation matrix
        # rotation_matrix = p.getMatrixFromQuaternion(orientation)
        # rotation_matrix = np.array(rotation_matrix).reshape(3, 3)

        # Define directions based on rotation
        x_direction = rotation_matrix[:, 0]
        y_direction = rotation_matrix[:, 1]
        z_direction = rotation_matrix[:, 2]
        
        # Draw X axis (red)
        self.client_id.addUserDebugLine(position,
                        position + length * x_direction,
                        lineColorRGB=[1, 0, 0],
                        lineWidth=width,
                        lifeTime=life_time)

        # Draw Y axis (green)
        self.client_id.addUserDebugLine(position,
                        position + length * y_direction,
                        lineColorRGB=[0, 1, 0],
                        lineWidth=width,
                        lifeTime=life_time)

        # Draw Z axis (blue)
        self.client_id.addUserDebugLine(position,
                        position + length * z_direction,
                        lineColorRGB=[0, 0, 1],
                        lineWidth=width,
                        lifeTime=life_time)
        

