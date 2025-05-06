import pybullet as p
import pybullet_data
import time
import os

physicsClient = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -9.81)
planeId = p.loadURDF("plane.urdf")

current_dir = os.path.dirname(os.path.abspath(__file__))
urdf_file_path = os.path.join(current_dir, "description", "mechanum.urdf")

print(urdf_file_path)

robot_start_pos = [0, 0, 0.5]
robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
use_fixed_base = False

print(f"Attempting to load URDF: {urdf_file_path}")

try:
    robot_id = p.loadURDF(
        urdf_file_path,
        basePosition=robot_start_pos,
        baseOrientation=robot_start_orientation,
        useFixedBase=use_fixed_base,
        flags=p.URDF_USE_INERTIA_FROM_FILE
    )
    print(f"Successfully loaded robot with ID: {robot_id}")

    num_joints = p.getNumJoints(robot_id)
    print(f"Number of joints: {num_joints}")
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        print(f"Joint {i}: {joint_info[1].decode('utf-8')}, Type: {joint_info[2]}")

    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=50,
        cameraPitch=-35,
        cameraTargetPosition=robot_start_pos
    )

    for i in range(10000):
        p.stepSimulation()
        time.sleep(1. / 240.)

except p.error as e:
    print(f"Error loading URDF: {e}")
    print("Common issues:")
    print("- URDF file not found at the specified path.")
    print("- URDF file has syntax errors.")
    print("- Mesh files referenced in the URDF are not found (paths in URDF are relative to the URDF file itself).")
    print("- Ensure all file paths in your URDF (for meshes, textures) are correct and accessible.")

finally:
    p.disconnect()