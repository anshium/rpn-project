import pybullet as p
import os
import math

from ball import spawn_sphere_with_velocity

class Robot:
    def __init__(self, urdf_path, base_position=[0, 0, 0], base_orientation_euler=[0, 0, 0],
                 use_fixed_base=False, physics_client_id=0,
                 # Mecanum specific parameters (provide if it's a mecanum robot)
                 mecanum_wheel_joint_names=None, # Dict: {"fl": "name1", "fr": "name2", "rl": "name3", "rr": "name4"}
                 wheel_radius=None,              # Radius of the mecanum wheels
                 half_wheelbase_lx=None,         # Half distance between front and rear axles (along robot's X)
                 half_track_width_ly=None          # Half distance between left and right wheels (along robot's Y)
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

        self.is_mecanum = False
        if mecanum_wheel_joint_names and wheel_radius is not None and \
           half_wheelbase_lx is not None and half_track_width_ly is not None:
            self.is_mecanum = True
            self.mecanum_wheel_joint_names_map = mecanum_wheel_joint_names # e.g. {"fl": "front_left_wheel_joint", ...}
            self.wheel_radius = wheel_radius
            self.half_wheelbase_lx = half_wheelbase_lx # a in some formulas
            self.half_track_width_ly = half_track_width_ly   # b in some formulas
            self.mecanum_joint_ids = {} # Stores pybullet joint IDs: {"fl": id1, "fr": id2, ...}
            self._map_mecanum_joints()
        elif mecanum_wheel_joint_names or wheel_radius or half_wheelbase_lx or half_track_width_ly:
            print("Warning: Mecanum parameters provided but some are missing. Mecanum drive will not be enabled.")


    def _load_robot(self):
        try:
            self.robot_id = p.loadURDF(
                self.urdf_path,
                basePosition=self.base_position,
                baseOrientation=self.base_orientation_quaternion,
                useFixedBase=self.use_fixed_base,
                flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION, # Added self-collision
                physicsClientId=self.physics_client_id
            )
            print(f"Successfully loaded robot '{os.path.basename(self.urdf_path)}' with ID: {self.robot_id}")
        except p.error as e:
            print(f"Error loading URDF '{self.urdf_path}': {e}")
            raise

    def _get_joint_info(self):
        if self.robot_id is None: return
        self.num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client_id)
        print(f"Robot has {self.num_joints} joints.")
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client_id)
            joint_id = info[0]
            joint_name = info[1].decode('utf-8')
            joint_type = info[2]
            link_name = info[12].decode('utf-8')
            self.joint_name_to_id[joint_name] = joint_id
            self.joint_id_to_name[joint_id] = joint_name
            self.link_name_to_id[link_name] = i
            print(f"  Joint {i}: ID={joint_id}, Name='{joint_name}', Type={joint_type}, Link='{link_name}'")
        self.link_name_to_id['base_link'] = -1

    def _map_mecanum_joints(self):
        if not self.is_mecanum: return
        required_keys = ["fl", "fr", "rl", "rr"]
        all_keys_found = True
        for key in required_keys:
            if key not in self.mecanum_wheel_joint_names_map:
                print(f"Error: Mecanum wheel key '{key}' not in mecanum_wheel_joint_names_map.")
                all_keys_found = False
                continue
            joint_name = self.mecanum_wheel_joint_names_map[key]
            if joint_name not in self.joint_name_to_id:
                print(f"Error: Mecanum joint name '{joint_name}' (for '{key}') not found in URDF joints.")
                all_keys_found = False
                continue
            self.mecanum_joint_ids[key] = self.joint_name_to_id[joint_name]

        if not all_keys_found or len(self.mecanum_joint_ids) != 4:
            print("Mecanum joint mapping failed. Disabling mecanum control.")
            self.is_mecanum = False
        else:
            print("Mecanum joints mapped successfully:")
            for key, jid in self.mecanum_joint_ids.items():
                print(f"  '{key}': Joint Name='{self.mecanum_wheel_joint_names_map[key]}', ID={jid}")

    def get_joint_id(self, joint_name):
        if joint_name not in self.joint_name_to_id:
            print(f"Warning: Joint '{joint_name}' not found. Available: {list(self.joint_name_to_id.keys())}")
            return None
        return self.joint_name_to_id[joint_name]

    def set_joint_position(self, joint_name, position, max_force=100.0, kp=0.1, kd=1.0):
        joint_id = self.get_joint_id(joint_name)
        if joint_id is not None and self.robot_id is not None:
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id, jointIndex=joint_id, controlMode=p.POSITION_CONTROL,
                targetPosition=position, force=max_force, positionGain=kp, velocityGain=kd,
                physicsClientId=self.physics_client_id
            )

    def set_mecanum_velocity(self, vx, vy, omega_z, max_force_per_wheel=10.0): # Added max_force_per_wheel here
        if not self.is_mecanum:
            print("Cannot set mecanum velocity: Robot not configured or initialized as mecanum.")
            return
        if self.robot_id is None: return

        # lx = self.half_wheelbase_lx (distance from CoG to front/rear axle)
        # ly = self.half_track_width_ly (distance from CoG to left/right wheel line)
        sum_lw = self.half_wheelbase_lx + self.half_track_width_ly

        # These are the desired linear velocities of the wheel's contact point with the ground
        # This specific set of equations is standard for a common Mecanum roller orientation (X-pattern)
        # FL rollers \, FR rollers /, RL rollers /, RR rollers \ (when viewed from top, rollers pointing towards center lines)
        # vx: positive forward
        # vy: positive strafe left
        # omega_z: positive CCW rotation
        target_wheel_linear_vels = {
            "fl": vx - vy - sum_lw * omega_z,
            "fr": vx + vy + sum_lw * omega_z,
            "rl": vx + vy - sum_lw * omega_z, # Note: Some conventions might swap vy or omega_z signs for rear wheels
            "rr": vx - vy + sum_lw * omega_z  # depending on specific roller setup and frame definitions.
        }                                     # This set is very common, though.

        joint_indices = []
        target_velocities_for_pb = []
        forces_for_pb = []

        # The order here defines the order in the arrays sent to PyBullet.
        # As long as mecanum_joint_ids[key] is correct, this is fine.
        ordered_keys = ["fl", "fr", "rl", "rr"]

        # print(f"Cmd: vx={vx}, vy={vy}, wz={omega_z}") # For debugging
        for key in ordered_keys:
            linear_vel_surface = target_wheel_linear_vels[key]
            angular_vel_raw = linear_vel_surface / self.wheel_radius

            # Apply compensation for URDF joint axis convention
            # This assumes that for "fr" and "rr" wheels, a positive joint velocity
            # in the URDF makes the wheel spin physically backward from robot's perspective.
            angular_vel_command = angular_vel_raw
            if key == "fr" or key == "rr":
                angular_vel_command = -angular_vel_raw
            
            # print(f"  {key}: lin_vel={linear_vel_surface:.2f}, ang_vel_raw={angular_vel_raw:.2f}, ang_vel_cmd={angular_vel_command:.2f}")

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

if __name__ == "__main__":
    import pybullet_data
    import time

    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    p.setPhysicsEngineParameter(numSolverIterations=150, physicsClientId=physics_client)


    p.setRealTimeSimulation(0)
    
    plane_id = p.loadURDF("plane.urdf", physicsClientId=physics_client)

    p.changeDynamics(plane_id, -1, lateralFriction=0.0, physicsClientId=physics_client)

    urdf_file = "/home/anshium/workspace/courses/rpn/project/description/mechanum.urdf"

    mecanum_joint_names = {
        "fl": "upper_left_wheel_joint",
        "fr": "upper_right_wheel_joint",
        "rl": "lower_left_wheel_joint",
        "rr": "lower_right_wheel_joint"
    }
    wheel_radius_val = 0.05
    half_wheelbase_lx_val = 0.150
    half_track_width_ly_val = 0.150

    try:
        robot = Robot(
            urdf_path=urdf_file,
            base_position=[0, 0, 0.1],
            physics_client_id=physics_client,
            mecanum_wheel_joint_names=mecanum_joint_names,
            wheel_radius=wheel_radius_val,
            half_wheelbase_lx=half_wheelbase_lx_val,
            half_track_width_ly=half_track_width_ly_val,
        )

        p.resetBaseVelocity(robot.robot_id,
                            linearVelocity=[1, 2, 0],
                            angularVelocity=[0, 0, 0],
                            physicsClientId=physics_client)
        
        sphere1_id = spawn_sphere_with_velocity(
            position=[-0.5, 0, 2],
            linear_velocity=[1, 0.5, 0],
            radius=0.05,
            mass=0.2,
            color=[1, 0, 0, 1], 
            # physics_client_id=robot.client_id
        )
        print(f"Spawned sphere 1 with ID: {sphere1_id}")

        test_duration = 20
        current_step = 0
        max_total_steps = 6 * test_duration

        while current_step < max_total_steps:
           
            p.resetBaseVelocity(robot.robot_id,
                            linearVelocity=[1, 0, 0],
                            angularVelocity=[0, 0, 0],
                            physicsClientId=robot.physics_client_id)

            p.stepSimulation(physicsClientId=physics_client)
            time.sleep(1./240.)
            current_step += 1

        time.sleep(10)

    except Exception as e:
        print(f"An error occurred during the test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'robot' in locals() and robot.robot_id is not None:
            robot.remove()
        p.disconnect(physicsClientId=physics_client)