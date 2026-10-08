import mujoco
import mujoco.viewer
import numpy as np
import time

# Load Skydio X2 Quadrotor
model_path = "mujoco_menagerie/skydio_x2/scene.xml"
try:
    model = mujoco.MjModel.from_xml_path(model_path)
except Exception:
    model = mujoco.MjModel.from_xml_path("skydio_x2/scene.xml")

data = mujoco.MjData(model)

# Identify drone chassis body ID
drone_body_names = ["x2", "base_link", "body", "chassis"]
body_id = 0
for name in drone_body_names:
    try:
        body_id = model.body(name).id
        break
    except KeyError:
        continue

# Calculate baseline hover force (m * g)
total_mass = np.sum(model.body_mass)
g_acc = np.linalg.norm(model.opt.gravity)
hover_force = total_mass * g_acc

# Flight Wrench Targets
f_z_cmd = hover_force
t_roll_cmd = 0.0
t_pitch_cmd = 0.0
t_yaw_cmd = 0.0

def keyboard_control(keycode):
    global f_z_cmd, t_roll_cmd, t_pitch_cmd, t_yaw_cmd
    # Pitch & Roll (Arrow Keys)
    if keycode == 265:     # UP Arrow: Pitch forward
        t_pitch_cmd = 0.04
    elif keycode == 264:   # DOWN Arrow: Pitch backward
        t_pitch_cmd = -0.04
    elif keycode == 263:   # LEFT Arrow: Roll left
        t_roll_cmd = -0.04
    elif keycode == 262:   # RIGHT Arrow: Roll right
        t_roll_cmd = 0.04
    # Altitude & Yaw (I, K, J, L Keys)
    elif keycode == 73:    # I Key: Climb (+fz)
        f_z_cmd += 0.4
    elif keycode == 75:    # K Key: Descend (-fz)
        f_z_cmd = max(0.0, f_z_cmd - 0.4)
    elif keycode == 74:    # J Key: Yaw spin left (+tz)
        t_yaw_cmd = 0.02
    elif keycode == 76:    # L Key: Yaw spin right (-tz)
        t_yaw_cmd = -0.02
    # Reset Target (Spacebar)
    elif keycode == 32:    # Spacebar: Stabilized Hover
        f_z_cmd = hover_force
        t_roll_cmd, t_pitch_cmd, t_yaw_cmd = 0.0, 0.0, 0.0

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
    print("CHALLENGE 2: SKYDIO X2 STABILIZED 6-DOF TELEOP")
    print("  Up / Down Arrows  : Pitch Forward / Backward")
    print("  Left / Right      : Roll Left / Right")
    print("  I / K Keys        : Climb / Descend")
    print("  J / L Keys        : Yaw Spin Left / Right")
    print("  Spacebar          : Return to Level Hover")
    print("========================================================\n")

    step_count = 0
    world_pos = np.zeros(3)
    world_mat = np.eye(3)

    while viewer.is_running():
        # Velocity damping for stable flight
        vel_lin = data.qvel[0:3]
        vel_ang = data.qvel[3:6]

        fx = -0.4 * vel_lin[0]
        fy = -0.4 * vel_lin[1]
        fz = f_z_cmd - 2.0 * vel_lin[2]

        tx = t_roll_cmd - 0.05 * vel_ang[0]
        ty = t_pitch_cmd - 0.05 * vel_ang[1]
        tz = t_yaw_cmd - 0.02 * vel_ang[2]

        data.xfrc_applied[body_id] = [fx, fy, fz, tx, ty, tz]

        mujoco.mj_step(model, data)

        drone_pos = data.xpos[body_id].copy()
        drone_mat = data.xmat[body_id].reshape(3, 3).copy()

        # Render 3D frame triads
        viewer.user_scn.ngeom = 0
        draw_axis(viewer.user_scn, world_pos, world_mat, size=0.5, width=0.012)
        draw_axis(viewer.user_scn, drone_pos, drone_mat, size=0.3, width=0.008)

        # Format matrix values for clean GUI presentation
        matrix_str = "\n".join(
            ["  ".join([f"{val:6.3f}" for val in row]) for row in drone_mat]
        )

        # Overlay Rotation Matrix inside top-left corner of MuJoCo 3D Window
        viewer.set_texts([
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                "3x3 Rotation Matrix (R_body_to_world):",
                matrix_str,
            )
        ])

        # Terminal printing every 15 steps
        if step_count % 15 == 0:
            print("\033[H\033[J", end="")
            print("========================================================")
            print(f" Quadrotor Pos (X, Y, Z) [m]: {np.round(drone_pos, 3)}")
            print("--------------------------------------------------------")
            print(" 3x3 Rotation Matrix (R_body_to_world):")
            print(np.round(drone_mat, 3))
            print("========================================================")

        step_count += 1
        viewer.sync()
        time.sleep(model.opt.timestep)
