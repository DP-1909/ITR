import mujoco
import mujoco.viewer
import numpy as np
import time

# Load TurtleBot3 Waffle Pi Model
model_path = "robotis_mujoco_menagerie/robotis_tb3/scene_turtlebot3_waffle_pi.xml"
try:
    model = mujoco.MjModel.from_xml_path(model_path)
except Exception:
    model = mujoco.MjModel.from_xml_path("robotis_tb3/scene_turtlebot3_waffle_pi.xml")

data = mujoco.MjData(model)

# Identify chassis body ID
CHASSIS_NAME = "base"
try:
    chassis_id = model.body(CHASSIS_NAME).id
except KeyError:
    chassis_id = model.body("base_link").id

# TurtleBot3 Physical Parameters
R = 0.033   # Wheel radius (m)
L = 0.287   # Wheelbase (m)

# Target velocity commands
v = 0.0  # Linear velocity (m/s)
w = 0.0  # Angular velocity (rad/s)

def keyboard_control(keycode):
    global v, w
    if keycode == 265:     # UP Arrow: Forward
        v += 0.05
    elif keycode == 264:   # DOWN Arrow: Reverse
        v -= 0.05
    elif keycode == 263:   # LEFT Arrow: Turn Left
        w += 0.2
    elif keycode == 262:   # RIGHT Arrow: Turn Right
        w -= 0.2
    elif keycode == 32:    # Spacebar: Emergency Stop
        v, w = 0.0, 0.0

def draw_axis(scn, pos, mat, size=0.4, width=0.01):
    colors = [
        [1, 0, 0, 1],  # X = Red
        [0, 1, 0, 1],  # Y = Green
        [0, 0, 1, 1],  # Z = Blue
    ]
    for axis in range(3):
        direction = mat[:, axis]
        end = pos + direction * size

        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3), np.zeros(3), np.zeros(9),
            np.array(colors[axis], dtype=np.float32)
        )
        mujoco.mjv_connector(
            g, mujoco.mjtGeom.mjGEOM_ARROW, width,
            pos, end
        )
        scn.ngeom += 1

with mujoco.viewer.launch_passive(model, data, key_callback=keyboard_control) as viewer:
    with viewer.lock():
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE

    print("\n========================================================")
    print("CHALLENGE 1: TURTLEBOT3 WAFFLE PI TELEOP & MATRIX DISPLAY")
    print("  Up / Down Arrows  : Linear Speed (+v / -v)")
    print("  Left / Right      : Angular Speed (+w / -w)")
    print("  Spacebar          : Emergency Stop")
    print("========================================================\n")

    step_count = 0
    world_pos = np.zeros(3)
    world_mat = np.eye(3)

    while viewer.is_running():
        # Differential drive kinematic velocity transformation
        w_left = (v - w * (L / 2.0)) / R
        w_right = (v + w * (L / 2.0)) / R

        # Send wheel angular velocities to actuators
        if model.nu >= 2:
            data.ctrl[0] = w_left
            data.ctrl[1] = w_right

        mujoco.mj_step(model, data)

        chassis_pos = data.xpos[chassis_id].copy()
        chassis_mat = data.xmat[chassis_id].reshape(3, 3).copy()

        # Render 3D frame triads (World origin and Chassis COM)
        viewer.user_scn.ngeom = 0
        draw_axis(viewer.user_scn, world_pos, world_mat, size=0.5, width=0.015)     # Fixed ground frame
        draw_axis(viewer.user_scn, chassis_pos, chassis_mat, size=0.3, width=0.01)  # Moving chassis frame

        # Format matrix values for clean GUI text presentation
        matrix_str = "\n".join(
            ["  ".join([f"{val:6.3f}" for val in row]) for row in chassis_mat]
        )

        # Overlay Rotation Matrix inside top-left corner of MuJoCo 3D Window
        viewer.set_texts([
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                "3x3 Rotation Matrix (R_chassis_to_world):",
                matrix_str,
            )
        ])

        # Terminal output streaming
        if step_count % 15 == 0:
            print("\033[H\033[J", end="")
            print("========================================================")
            print(f" Linear Vel (v): {v:.2f} m/s | Angular Vel (w): {w:.2f} rad/s")
            print(f" Chassis Position (X, Y, Z) [m]: {np.round(chassis_pos, 3)}")
            print("--------------------------------------------------------")
            print(" 3x3 Rotation Matrix (R_chassis_to_world):")
            print(np.round(chassis_mat, 3))
            print("========================================================")

        step_count += 1
        viewer.sync()
        time.sleep(model.opt.timestep)
