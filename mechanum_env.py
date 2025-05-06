import pybullet as p
import os
import math
import numpy as np
import time # For sleeps and timing

# Import from ball.py
from ball import spawn_sphere_with_velocity, predict_landing_point

# Import the MPC controller
from mpc_controller import MPCController 

class Robot:
    def __init__(self, urdf_path, base_position=[0, 0, 0], base_orientation_euler=[0, 0, 0],
                 use_fixed_base=False, physics_client_id=0,
                 mecanum_wheel_joint_names=None, 
                 wheel_radius=None,              
                 half_wheelbase_lx=None,         
                 half_track_width_ly=None       
                 ):
        self.urdf_path = urdf_path
        self.base_position = base_position 
        self.base_orientation_quaternion = p.getQuaternionFromEuler(base_orientation_euler)
        self.use_fixed_base = use_fixed_base
        self.physics_client_id = physics_client_id

        self.robot_id = None
        self.num_joints = 0
        self.joint_name_to_id = {}
        self.joint_id_to_name = {}
        self.link_name_to_id = {}

        self._load_robot()
        self._get_joint_info()

        self.is_mecanum = False # This flag is still set based on params, even if not used by this specific MPC mode
        if mecanum_wheel_joint_names and wheel_radius is not None and \
           half_wheelbase_lx is not None and half_track_width_ly is not None:
            self.is_mecanum = True
            self.mecanum_wheel_joint_names_map = mecanum_wheel_joint_names
            self.wheel_radius = wheel_radius
            self.half_wheelbase_lx = half_wheelbase_lx 
            self.half_track_width_ly = half_track_width_ly  
            self.mecanum_joint_ids = {} 
            self._map_mecanum_joints()
        elif mecanum_wheel_joint_names or wheel_radius or half_wheelbase_lx or half_track_width_ly:
            print("Warning: Mecanum parameters provided but some are missing. Mecanum drive (if used elsewhere) might not be enabled.")


    def _load_robot(self):
        try:
            self.robot_id = p.loadURDF(
                self.urdf_path,
                basePosition=self.base_position,
                baseOrientation=self.base_orientation_quaternion,
                useFixedBase=self.use_fixed_base,
                flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION,
                physicsClientId=self.physics_client_id
            )
            print(f"Successfully loaded robot '{os.path.basename(self.urdf_path)}' with ID: {self.robot_id}")
        except p.error as e:
            print(f"Error loading URDF '{self.urdf_path}': {e}")
            raise

    def _get_joint_info(self):
        if self.robot_id is None: return
        self.num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client_id)
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client_id)
            joint_id = info[0]
            joint_name = info[1].decode('utf-8')
            link_name = info[12].decode('utf-8')
            self.joint_name_to_id[joint_name] = joint_id
            self.joint_id_to_name[joint_id] = joint_name
            self.link_name_to_id[link_name] = i 
        self.link_name_to_id['base_link'] = -1 

    def _map_mecanum_joints(self):
        if not self.is_mecanum: return # Only map if configured to be mecanum
        required_keys = ["fl", "fr", "rl", "rr"]
        all_keys_found = True
        for key in required_keys:
            if key not in self.mecanum_wheel_joint_names_map:
                print(f"Error: Mecanum wheel key '{key}' not in mecanum_wheel_joint_names_map.")
                all_keys_found = False; continue
            joint_name = self.mecanum_wheel_joint_names_map[key]
            if joint_name not in self.joint_name_to_id:
                print(f"Error: Mecanum joint name '{joint_name}' (for '{key}') not found in URDF joints.")
                all_keys_found = False; continue
            self.mecanum_joint_ids[key] = self.joint_name_to_id[joint_name]

        if not all_keys_found or len(self.mecanum_joint_ids) != 4:
            print("Mecanum joint mapping failed. Mecanum motor control will not be available.")
            self.is_mecanum = False # Set to false if mapping fails
        else:
            print("Mecanum joints mapped successfully (for potential motor control).")

    # This method remains for potential use, but MPC will use direct velocity control
    def set_mecanum_velocity(self, vx, vy, omega_z, max_force_per_wheel=10.0):
        if not self.is_mecanum or self.robot_id is None:
            # print("Cannot set mecanum motor velocity: Robot not configured as mecanum or no robot_id.")
            return

        sum_lw = self.half_wheelbase_lx + self.half_track_width_ly
        target_wheel_linear_vels = {
            "fl": vx - vy - sum_lw * omega_z,
            "fr": vx + vy + sum_lw * omega_z,
            "rl": vx + vy - sum_lw * omega_z, 
            "rr": vx - vy + sum_lw * omega_z  
        }                                     

        joint_indices = []
        target_velocities_for_pb = []
        forces_for_pb = []
        ordered_keys = ["fl", "fr", "rl", "rr"] 

        for key in ordered_keys:
            linear_vel_surface = target_wheel_linear_vels[key]
            angular_vel_raw = linear_vel_surface / self.wheel_radius
            angular_vel_command = angular_vel_raw
            if key == "fr" or key == "rr": 
                angular_vel_command = -angular_vel_raw
            
            joint_indices.append(self.mecanum_joint_ids[key])
            target_velocities_for_pb.append(angular_vel_command)
            forces_for_pb.append(max_force_per_wheel)

        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=joint_indices,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=target_velocities_for_pb,
            forces=forces_for_pb,
            physicsClientId=self.physics_client_id
        )

    def remove(self):
        if self.robot_id is not None:
            try:
                p.removeBody(self.robot_id, physicsClientId=self.physics_client_id)
                self.robot_id = None
                print("Robot removed.")
            except p.error as e:
                print(f"Error removing robot: {e}")


    def get_2d_pose(self):
        if self.robot_id is None: return None, None
        try:
            pos, orn_quat = p.getBasePositionAndOrientation(self.robot_id, physicsClientId=self.physics_client_id)
            _, _, yaw = p.getEulerFromQuaternion(orn_quat)
            return np.array([pos[0], pos[1]]), yaw
        except p.error as e:
            print(f"Error getting robot pose: {e}")
            return None, None


    def run_mpc_navigation(self, goal_xy, obstacles_pybullet_info, mpc_config, 
                           max_mpc_steps=1000, goal_threshold=0.2, viz_mpc_path=False,
                           pb_time_step=1./240.):
        # Removed: if not self.is_mecanum ... as direct velocity control doesn't need it.
        # However, robot_effective_radius might still use mecanum params if available.
        if self.is_mecanum: # Use mecanum params for radius if available
            robot_effective_radius = np.sqrt(self.half_wheelbase_lx**2 + self.half_track_width_ly**2) + 0.05
        else: # Fallback if not mecanum (e.g., a generic robot base)
            robot_effective_radius = 0.2 # Default generic radius
            print("Warning: Robot not configured as mecanum. Using default effective radius for MPC.")
        
        initial_vx_guess = np.zeros(mpc_config['planning_horizon'])
        initial_vy_guess = np.zeros(mpc_config['planning_horizon'])

        mpc_controller = MPCController(
            robot_radius=robot_effective_radius,
            vxmax=mpc_config['vxmax'], vymax=mpc_config['vymax'],
            planning_horizon=mpc_config['planning_horizon'],
            control_horizon=mpc_config['control_horizon'],
            del_t=mpc_config['del_t'],
            initial_vx_guess=initial_vx_guess, initial_vy_guess=initial_vy_guess
        )
        
        mpc_path_line_id = None
        goal_xy_np = np.array(goal_xy)

        # Fetch initial pose once before the main MPC loop
        current_pos_xy, current_yaw = self.get_2d_pose()
        if current_pos_xy is None:
            print("Failed to get initial robot pose. Stopping MPC.")
            return False

        for mpc_step_count in range(max_mpc_steps):
            # Goal check uses current_pos_xy from the end of the previous MPC execution
            if np.linalg.norm(current_pos_xy - goal_xy_np) < goal_threshold:
                print(f"Goal reached in {mpc_step_count} MPC steps!")
                p.resetBaseVelocity(self.robot_id, [0,0,0], [0,0,0], physicsClientId=self.physics_client_id) # Stop
                return True

            obstacles_info_for_mpc = []
            for obs_id, obs_radius in obstacles_pybullet_info: 
                try:
                    obs_pos, _ = p.getBasePositionAndOrientation(obs_id, physicsClientId=self.physics_client_id)
                    obstacles_info_for_mpc.append({'position': [obs_pos[0], obs_pos[1]], 'radius': obs_radius})
                except p.error:
                    print(f"Warning: Could not get info for obstacle ID {obs_id}. Skipping.")

            # MPC plans using current_pos_xy and current_yaw
            vx_world_cmds, vy_world_cmds = mpc_controller.compute_control_actions(
                current_pos=current_pos_xy, goal_pos=goal_xy_np, obstacles_info=obstacles_info_for_mpc
            )
            
            if vx_world_cmds is None or vy_world_cmds is None: 
                print(f"MPC failed to find a solution at step {mpc_step_count}. Stopping.")
                p.resetBaseVelocity(self.robot_id, [0,0,0], [0,0,0], physicsClientId=self.physics_client_id) # Stop
                return False

            if viz_mpc_path and mpc_controller.optimized_vx is not None : 
                if mpc_path_line_id is not None:
                    p.removeUserDebugItem(mpc_path_line_id, physicsClientId=self.physics_client_id)
                mpc_planned_path_points = []
                temp_x, temp_y = current_pos_xy[0], current_pos_xy[1] # Start path from current MPC planning pos
                debug_line_z = self.base_position[2] if self.base_position is not None and len(self.base_position) > 2 else 0.05
                mpc_planned_path_points.append([temp_x, temp_y, debug_line_z])
                for i in range(mpc_controller.planning_horizon):
                    temp_x += mpc_controller.optimized_vx[i] * mpc_controller.del_t
                    temp_y += mpc_controller.optimized_vy[i] * mpc_controller.del_t
                    mpc_planned_path_points.append([temp_x, temp_y, debug_line_z])
                if len(mpc_planned_path_points) > 1:
                    # mpc_path_line_id = p.addUserDebugLine(
                    #     [pt[:3] for pt in mpc_planned_path_points[:-1]],  
                    #     [pt[:3] for pt in mpc_planned_path_points[1:]],  
                    #     lineColorRGB=[1, 0, 0], 
                    #     lineWidth=2, 
                    #     physicsClientId=self.physics_client_id
                    # )
                    pass

            # Execute the control horizon
            for i in range(mpc_config['control_horizon']):
                vx_w, vy_w = vx_world_cmds[i], vy_world_cmds[i]
                
                # current_yaw here is from the end of the previous del_t interval's simulation
                # or the initial yaw if i=0 for this MPC cycle.
                omega_z_cmd = 0.0
                if np.linalg.norm([vx_w, vy_w]) > 0.05: 
                    target_yaw_for_step = np.arctan2(vy_w, vx_w)
                    angle_diff_for_step = target_yaw_for_step - current_yaw 
                    angle_diff_for_step = (angle_diff_for_step + np.pi) % (2 * np.pi) - np.pi 
                    omega_z_cmd = 2.0 * angle_diff_for_step # P-control for yaw. Tune K_omega
                    omega_z_cmd = np.clip(omega_z_cmd, -1.5, 1.5) # Clamp omega_z

                # Apply desired world-frame velocity directly to the robot's base
                p.resetBaseVelocity(
                    self.robot_id,
                    linearVelocity=[vx_w * 2, vy_w * 2, 0], # Z velocity is zero
                    angularVelocity=[0, 0, omega_z_cmd],
                    physicsClientId=self.physics_client_id
                )

                # Simulate for del_t
                num_pb_steps = max(1, int(mpc_config['del_t'] / pb_time_step))
                for _ in range(num_pb_steps):
                    p.stepSimulation(physicsClientId=self.physics_client_id)
                    if p.getConnectionInfo(physicsClientId=self.physics_client_id)['connectionMethod'] == p.GUI:
                        time.sleep(pb_time_step) 
                
                # Update current_pos_xy and current_yaw for the next iteration of the control_horizon loop
                # OR for the next MPC planning cycle if this is the last step of control_horizon.
                current_pos_xy_step, current_yaw_step = self.get_2d_pose() 
                if current_yaw_step is not None:
                    current_yaw = current_yaw_step 
                if current_pos_xy_step is not None:
                    current_pos_xy = current_pos_xy_step
                else: # Failed to get pose, critical error
                    print("CRITICAL: Failed to get robot pose during control horizon. Stopping.")
                    p.resetBaseVelocity(self.robot_id, [0,0,0], [0,0,0], physicsClientId=self.physics_client_id)
                    return False
            
            # current_pos_xy and current_yaw are now updated after the full control_horizon execution
            print(f"MPC Step {mpc_step_count}: Pos: [{current_pos_xy[0]:.2f}, {current_pos_xy[1]:.2f}], Yaw: {current_yaw:.2f}, Goal: {goal_xy}")


        print("Max MPC steps reached, goal not achieved.")
        p.resetBaseVelocity(self.robot_id, [0,0,0], [0,0,0], physicsClientId=self.physics_client_id) # Stop
        return False


if __name__ == "__main__":
    import pybullet_data
    import random

    random.seed(10)

    physics_client = p.connect(p.GUI) 
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    sim_gravity_z = -1.81
    p.setGravity(0, 0, sim_gravity_z, physicsClientId=physics_client)
    p.setRealTimeSimulation(0) 
    
    pb_sim_time_step = 1./240.
    p.setTimeStep(pb_sim_time_step, physicsClientId=physics_client)

    plane_id = p.loadURDF("plane.urdf", physicsClientId=physics_client)

    # --- Robot Configuration ---
    urdf_file = "/home/anshium/workspace/courses/rpn/project/description/mechanum.urdf" 
    if not os.path.exists(urdf_file):
        print(f"FATAL ERROR: URDF file not found at {urdf_file}")
        print("Please update the urdf_file variable in the __main__ section.")
        p.disconnect(physicsClientId=physics_client)
        exit()

    mecanum_joint_names_map = { # These will be used for 'is_mecanum' and radius calculation if present
        "fl": "upper_left_wheel_joint", "fr": "upper_right_wheel_joint",
        "rl": "lower_left_wheel_joint", "rr": "lower_right_wheel_joint"
    }
    robot_wheel_radius = 0.05      
    robot_half_wheelbase_lx = 0.15 
    robot_half_track_width_ly = 0.15 
    
    robot_start_position = [0, 0, 0.06] 

    robot_instance = Robot(
        urdf_path=urdf_file,
        base_position=robot_start_position,
        physics_client_id=physics_client,
        mecanum_wheel_joint_names=mecanum_joint_names_map, # Provide for radius calc, even if not using motors
        wheel_radius=robot_wheel_radius,
        half_wheelbase_lx=robot_half_wheelbase_lx,
        half_track_width_ly=robot_half_track_width_ly,
    )

    # --- Ball Configuration ---
    ball_initial_pos = [-2.0, 1.0, 2.0] 
    ball_initial_vel = [1, random.random() * 2, random.random() * 2 ]   
    ball_radius = 0.1
    ball_mass = 0.2
    ball_color = [0.8, 0.5, 0.2, 1] 

    ball_id = spawn_sphere_with_velocity(
        position=ball_initial_pos,
        linear_velocity=ball_initial_vel,
        radius=ball_radius,
        mass=ball_mass,
        color=ball_color,
        physics_client_id=physics_client
    )
    print(f"Spawned ball with ID: {ball_id}")

    # --- Predict Ball Landing and Set as Goal ---
    ground_contact_z = 0.0 
    landing_x, landing_y, _, time_to_land = predict_landing_point(
        initial_position=ball_initial_pos,
        initial_velocity=ball_initial_vel,
        gravity_z=sim_gravity_z,
        ground_z=ground_contact_z
    )

    target_goal_xy = None
    if landing_x is not None and landing_y is not None:
        target_goal_xy = [landing_x, landing_y]
        print(f"Predicted ball landing at: ({landing_x:.2f}, {landing_y:.2f}) in {time_to_land:.2f}s. Setting as robot goal.")
        p.createMultiBody(baseMass=0, 
                          baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0.8, 0.5, 0.2, 0.5]),
                          basePosition=[landing_x, landing_y, ground_contact_z + 0.025],
                          physicsClientId=physics_client)
        goal_marker_radius = 0.1
        p.createMultiBody(baseMass=0, 
                          baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=goal_marker_radius, rgbaColor=[0, 1, 0, 0.7]),
                          basePosition=[target_goal_xy[0], target_goal_xy[1], ground_contact_z + goal_marker_radius],
                          physicsClientId=physics_client)
    else:
        print("Ball is not predicted to land on the ground (or prediction failed). Robot will not navigate.")


    # --- Obstacle Configuration ---
    obstacles_in_pybullet = [] 
    obstacle_definitions = [
        ([-0.5, 1.5, 0.25], 0.10), 
        ([1.0, -1.0, 0.15], 0.05),
        ([1.5, 2.0, 0.3], 0.15)      
    ]
    for pos, rad in obstacle_definitions:
        col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=rad, physicsClientId=physics_client)
        vis_shape = p.createVisualShape(p.GEOM_SPHERE, radius=rad, rgbaColor=[0.8, 0.2, 0.2, 1], physicsClientId=physics_client)
        obs_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape,
                                   baseVisualShapeIndex=vis_shape, basePosition=pos,
                                   physicsClientId=physics_client)
        obstacles_in_pybullet.append((obs_id, rad))

    # --- MPC Configuration ---
    mpc_settings = {
        'planning_horizon': 20,  
        'control_horizon': 5,    
        'del_t': 0.1,            
        'vxmax': 100, # Max world X-velocity (m/s) - might need adjustment           
        'vymax': 100  # Max world Y-velocity (m/s) - might need adjustment
    }
    
    print(f"Robot start: {robot_start_position[:2]}")
    
    try:
        if robot_instance.robot_id is None:
            print("Robot not loaded. Exiting.")
        elif target_goal_xy is not None: 
            print(f"Starting MPC navigation (direct velocity) to predicted ball landing: {target_goal_xy}")
            success = robot_instance.run_mpc_navigation(
                goal_xy=target_goal_xy,
                obstacles_pybullet_info=obstacles_in_pybullet,
                mpc_config=mpc_settings,
                max_mpc_steps=300, 
                goal_threshold=0.25, 
                viz_mpc_path=True,
                pb_time_step=pb_sim_time_step
            )
            if success: print("MPC Navigation Succeeded!")
            else: print("MPC Navigation Failed or Timed Out.")
        else:
            print("No valid landing point for robot. Simulating ball flight for a few seconds...")
            for _ in range(int(5 / pb_sim_time_step)): 
                 p.stepSimulation(physicsClientId=physics_client)
                 if p.getConnectionInfo(physicsClientId=physics_client)['connectionMethod'] == p.GUI:
                        time.sleep(pb_sim_time_step)

    except Exception as e_main:
        print(f"An error occurred during simulation: {e_main}")
        import traceback
        traceback.print_exc()
    finally:
        print("Simulation finished. Press Enter in the terminal to close PyBullet.")
        input() 
        if robot_instance and robot_instance.robot_id is not None:
            robot_instance.remove()
        p.disconnect(physicsClientId=physics_client)