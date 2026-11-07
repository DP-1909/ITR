# ------------------------------------------------------------------------------
# This script constructs a composite MuJoCo model by attaching a Robotiq 2F-85 
# gripper to a single-arm HEAL robot using MJCF files. The combined model is 
# then visualized using MuJoCo's passive viewer, with gravity compensation applied.
#
# The robot and gripper XML files are loaded using **relative paths** resolved 
# from the script location, making this script portable and independent of the 
# machine-specific absolute paths. You can run it on **any PC** as long as the 
# folder structure within the repository remains consistent.
#
# Gravity compensation is handled using the feedforward bias force term
# `qfrc_bias`, ensuring the robot stays in equilibrium during passive viewing.
# So you wouldn't see the robot falling in simulation since the gravity compensation toreques are applied.
#
# To run this, ensure the following directory structure exists:
#     └── robot_descriptions/
#         ├── single_arm_heal_effort_actuation_rs_mj.xml
#         └── robotiq_2f85_v4/2f85.xml
# ------------------------------------------------------------------------------

from pathlib import Path
import mujoco
import mujoco.viewer

def construct_heal_with_gripper():
    """
    Constructs a MuJoCo model by attaching a Robotiq 2F-85 gripper to the
    HEAL single-arm robot. The attachment is done at the 'right_center' site.
    """
    # Resolve assets from the script's repository location, not the current
    # working directory or a machine-specific absolute path.
    repo_root = Path(__file__).resolve().parent.parent
    desc_dir = repo_root / "robot_descriptions"
    arm_path = desc_dir / "single_arm_heal_effort_actuation_rs_mj.xml"
    grip_path = desc_dir / "robotiq_2f85_v4" / "2f85.xml"

    missing = [path for path in (arm_path, grip_path) if not path.is_file()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing required MuJoCo asset(s):\n"
            f"{missing_text}\n"
            f"Expected assets under: {desc_dir}"
        )

    # Load the MJCF specs for the arm and the gripper
    arm = mujoco.MjSpec.from_file(str(arm_path))
    grip = mujoco.MjSpec.from_file(str(grip_path))

    # Attach the gripper to the HEAL arm at the specified site with a prefix
    arm.attach(grip, prefix='gripper/', site=arm.site('right_center'))

    # Compile and return the final combined model
    return arm.compile()

def compensate_gravity(model, data):
    """
    Applies feedforward control to each actuator to compensate for gravity and Coriolis effects.
    This uses the bias force vector `data.qfrc_bias`, which is populated by `mj_forward()`.
    """
    for aid in range(model.nu):
        joint_dof = model.actuator_trnid[aid][0]
        data.ctrl[aid] = data.qfrc_bias[joint_dof]

if __name__ == '__main__':
    # Build the combined HEAL + gripper model
    model = construct_heal_with_gripper()
    data  = mujoco.MjData(model)
    print(data.qpos)

    # Compute bias forces for the initial configuration
    mujoco.mj_forward(model, data)

    # Launch the passive MuJoCo viewer to visualize simulation without user control
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Continuously run the simulation with gravity compensation until the viewer is closed
        while viewer.is_running():
            compensate_gravity(model, data)
            mujoco.mj_step(model, data)
            viewer.sync()
