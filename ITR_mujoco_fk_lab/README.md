# ITR MuJoCo Forward Kinematics Lab (`ITR_mujoco_fk_lab`)

This folder contains the MuJoCo Forward Kinematics (FK) laboratory assignments and robot descriptions.

## Overview
- **`spawn_franka.py`**: Franka Emika Panda (7-DOF) interactive simulation featuring:
  - Tkinter slider GUI (joint angles + gripper control)
  - Analytical Forward Kinematics computation
  - Comparison between Analytical FK and MuJoCo simulation coordinates (real-time error analysis in mm and deg)
  - Jacobian computation and velocity tracking
- **`spawn_heal.py`**: Addverb HEAL (6-DOF) interactive simulation featuring analytical FK and real-time telemetry HUD
- **`robot_descriptions/`**: Robot models and meshes:
  - Franka Emika Panda (`franka/`, `franka_clean.xml`)
  - Addverb HEAL (`heal_meshes/`, `single_arm_heal_effort_actuation_rs.xml`)
  - UR5 (`ur5/`, `ur5.xml`, `dual_ur5.xml`)
  - Grippers (Robotiq 2F-85 / 85)
  - Unitree G1 (`unitree_g1/`)
  - Arenas, cylinders, and plate models

## Setup & Requirements

### 1. Using Conda
```bash
conda env create -f environment.yml
conda activate itr_fk
```

### 2. Using Pip
```bash
pip install -r requirements.txt
```

## Running the Simulations

You can run the simulation scripts either from the repository root or from inside this directory:

### Franka Emika Panda (7-DOF)
```bash
python3 spawn_franka.py
```
or from the repository root:
```bash
python3 ITR_mujoco_fk_lab/spawn_franka.py
```

### Addverb HEAL (6-DOF)
```bash
python3 spawn_heal.py
```
or from the repository root:
```bash
python3 ITR_mujoco_fk_lab/spawn_heal.py
```
