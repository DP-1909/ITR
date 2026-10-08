# ITR MuJoCo Simulations

## Setup

Use Python 3 and install the simulation dependencies:

```bash
python3 -m pip install mujoco numpy
```

Each simulation also needs its corresponding model files:

- Quadrotor: `mujoco_menagerie/skydio_x2/scene.xml`
- Waffle Pi: `robotis_mujoco_menagerie/robotis_tb3/scene_turtlebot3_waffle_pi.xml`

The scripts search for these model directories relative to the repository and
in the home directory, so they can be started from any working directory. If
the model is stored elsewhere, place its model directory in one of those
locations before running the script.

## Run

From the repository root:

```bash
python3 "Lab 1/Quadrotor.py"
python3 "Lab 1/Waffle_Pi.py"
```

Both programs open a MuJoCo viewer. Close the viewer window to exit.

### Quadrotor controls

- Arrow keys: forward/back and strafe left/right
- `W` / `S`: ascend / descend
- `A` / `D`: yaw left / right
- `Space`: hold position
- `R`: reset
- `F`: toggle camera follow

### Waffle Pi controls

- Up / Down arrows: increase / decrease linear speed
- Left / Right arrows: turn left / right
- `Space`: emergency stop
