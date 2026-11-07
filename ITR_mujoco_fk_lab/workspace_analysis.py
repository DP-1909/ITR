"""
workspace_analysis.py — Task vs Dexterous Workspace Analysis
=============================================================
Computes and visualizes two workspaces for the Franka Panda (7-DOF)
and Addverb HEAL (6-DOF) robots:

  1. Task Workspace:
     All reachable TCP (tool-center-point) positions via Monte Carlo
     random joint sampling within joint limits.

  2. Dexterous Workspace:
     Subset of task workspace where the manipulator can achieve
     arbitrary end-effector yaw orientations. For each candidate
     TCP position, we test multiple yaw angles and verify the
     manipulator can reach them all (checking via Jacobian rank).

Outputs:
  - 3D interactive scatter plots (matplotlib)
  - 2D XY occupancy map with table region overlay
  - Saved PNG figures in the output directory

Usage:
  cd ITR_mujoco_fk_lab && python3 workspace_analysis.py [--robot franka|heal] [--samples N]

Dependencies:
  numpy, scipy, matplotlib
"""

import os
import sys
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
import matplotlib.cm as cm


# ═══════════════════════════════════════════════════════════════════
# 1. FK MATH UTILITIES
# ═══════════════════════════════════════════════════════════════════

def quat_to_rotmat(q):
    """Quaternion [w, x, y, z] → 3x3 rotation matrix."""
    q = np.array(q, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x*x + y*y)]
    ])


def euler_xyz_to_rotmat(e):
    """MuJoCo intrinsic xyz euler → 3x3 rotation matrix: R = Rx(e0) @ Ry(e1) @ Rz(e2)."""
    cx, sx = np.cos(e[0]), np.sin(e[0])
    cy, sy = np.cos(e[1]), np.sin(e[1])
    cz, sz = np.cos(e[2]), np.sin(e[2])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def rot_axis(axis, angle):
    """Rodrigues' formula: rotation about an arbitrary axis by angle (rad)."""
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def make_transform(pos, rot3x3):
    """Build a 4×4 homogeneous transformation from position + rotation."""
    T = np.eye(4)
    T[:3, :3] = rot3x3
    T[:3, 3] = pos
    return T


# ═══════════════════════════════════════════════════════════════════
# 2. FRANKA PANDA (7-DOF) KINEMATIC CHAIN
# ═══════════════════════════════════════════════════════════════════

FRANKA_CHAIN = [
    # (translation, orientation_quat, joint_axis_or_None)
    ([0.0, 0.0, 0.0],     [1, 0, 0, 0],                    None),       # link0 (base, fixed)
    ([0, 0, 0.333],        [1, 0, 0, 0],                    [0, 0, 1]),  # link1 → joint1
    ([0, 0, 0],            [1, -1, 0, 0],                   [0, 0, 1]),  # link2 → joint2
    ([0, -0.316, 0],       [1, 1, 0, 0],                    [0, 0, 1]),  # link3 → joint3
    ([0.0825, 0, 0],       [1, 1, 0, 0],                    [0, 0, 1]),  # link4 → joint4
    ([-0.0825, 0.384, 0],  [1, -1, 0, 0],                   [0, 0, 1]),  # link5 → joint5
    ([0, 0, 0],            [1, 1, 0, 0],                    [0, 0, 1]),  # link6 → joint6
    ([0.088, 0, 0],        [1, 1, 0, 0],                    [0, 0, 1]),  # link7 → joint7
    ([0, 0, 0.107],        [0.9238795, 0, 0, -0.3826834],   None),       # hand (fixed flange)
]

# Joint limits from franka/mjx_panda.xml
FRANKA_JOINT_LIMITS = [
    (-2.8973, 2.8973),    # joint1 (default range)
    (-1.7628, 1.7628),    # joint2
    (-2.8973, 2.8973),    # joint3
    (-3.0718, -0.0698),   # joint4
    (-2.8973, 2.8973),    # joint5
    (-0.0175, 3.7525),    # joint6
    (-2.8973, 2.8973),    # joint7
]


def analytical_fk_franka(q):
    """Compute 4×4 homogeneous transform of the Franka end-effector (hand body)."""
    T = np.eye(4)
    joint_idx = 0
    for (pos, quat, axis) in FRANKA_CHAIN:
        R_body = quat_to_rotmat(quat)
        T_body = make_transform(pos, R_body)
        T = T @ T_body
        if axis is not None:
            R_joint = rot_axis(axis, q[joint_idx])
            T = T @ make_transform([0, 0, 0], R_joint)
            joint_idx += 1
    return T


# ═══════════════════════════════════════════════════════════════════
# 3. ADDVERB HEAL (6-DOF) KINEMATIC CHAIN
# ═══════════════════════════════════════════════════════════════════

# Precompute rotation matrices for HEAL chain
_HEAL_R_LINK2 = quat_to_rotmat([0.707105, 0.707108, 0, 0])
_HEAL_R_LINK3 = euler_xyz_to_rotmat([0, 0, -1.57])
_HEAL_R_LINK4 = euler_xyz_to_rotmat([-1.57, 0, 0])
_HEAL_R_LINK5 = euler_xyz_to_rotmat([0.50951, 0, 1.57])
_HEAL_R_EE    = quat_to_rotmat([0.707105, 0.707108, 0, 0])

HEAL_CHAIN = [
    # (translation, rotation_3x3, joint_axis_or_None)
    ([0, 0, 0],              np.eye(3),        None),           # base_link (fixed)
    ([0, 0, 0.171],          np.eye(3),        [0, 0, 1]),      # link_1 → joint_1
    ([0, 0.0875, 0.1498],    _HEAL_R_LINK2,    [0, 0, -1]),     # link_2 → joint_2
    ([0, 0.3, 0],            _HEAL_R_LINK3,    [0, 0, 1]),      # link_3 → joint_3
    ([0, 0.1593, 0.0875],    _HEAL_R_LINK4,    [0, 0, 1]),      # link_4 → joint_4
    ([0, 0.03185, 0.16105],  _HEAL_R_LINK5,    [0, 0, 1]),      # link_5 → joint_5
    ([0, -0.1227, 0.0654],   _HEAL_R_EE,       [0, 0, -1]),     # end_effector → joint_6
]

# Joint limits from single_arm_heal_effort_actuation_rs.xml
HEAL_JOINT_LIMITS = [
    (-3.14159, 3.14159),  # joint_1
    (-3.14159, 3.14159),  # joint_2
    (-3.14159, 3.14159),  # joint_3
    (-3.14159, 3.14159),  # joint_4
    (-3.14159, 3.14159),  # joint_5
    (-3.14159, 3.14159),  # joint_6
]


def analytical_fk_heal(q):
    """Compute 4×4 homogeneous transform of the HEAL end-effector."""
    T = np.eye(4)
    joint_idx = 0
    for (pos, R_body, axis) in HEAL_CHAIN:
        T_body = make_transform(pos, R_body)
        T = T @ T_body
        if axis is not None:
            R_joint = rot_axis(axis, q[joint_idx])
            T = T @ make_transform([0, 0, 0], R_joint)
            joint_idx += 1
    return T


# ═══════════════════════════════════════════════════════════════════
# 4. WORKSPACE COMPUTATION
# ═══════════════════════════════════════════════════════════════════

def compute_task_workspace(fk_func, joint_limits, num_samples=50000, seed=42):
    """
    Compute the task workspace by Monte Carlo sampling of random joint
    configurations within joint limits.

    Returns:
        positions: (N, 3) array of reachable TCP positions.
        orientations: (N, 3, 3) array of corresponding rotation matrices.
    """
    rng = np.random.default_rng(seed)
    n_joints = len(joint_limits)
    limits = np.array(joint_limits)

    # Generate random joint angles uniformly within limits
    q_samples = rng.uniform(limits[:, 0], limits[:, 1], size=(num_samples, n_joints))

    positions = np.zeros((num_samples, 3))
    orientations = np.zeros((num_samples, 3, 3))

    for i in range(num_samples):
        T = fk_func(q_samples[i])
        positions[i] = T[:3, 3]
        orientations[i] = T[:3, :3]

    return positions, orientations


def compute_dexterous_workspace(fk_func, joint_limits, task_positions,
                                 grid_resolution=0.04, n_yaw_tests=8,
                                 n_ik_attempts=50, seed=123):
    """
    Compute the dexterous workspace: the subset of the task workspace
    where the end-effector can achieve arbitrary yaw orientations.

    Strategy:
      1. Voxelize the task workspace into a 3D grid.
      2. For each occupied voxel, test if multiple different yaw angles
         can be reached by sampling random configurations and checking
         if any config places the TCP near the voxel center with the
         desired yaw orientation.

    Args:
        fk_func: Forward kinematics function.
        joint_limits: List of (lo, hi) joint limits.
        task_positions: (N, 3) array of task workspace positions.
        grid_resolution: Voxel size in meters.
        n_yaw_tests: Number of yaw orientations to test (evenly spaced).
        n_ik_attempts: Random configs to try per voxel per yaw.
        seed: Random seed.

    Returns:
        dexterous_centers: (M, 3) array of dexterous voxel centers.
        dex_scores: (M,) fraction of yaw angles achieved per voxel.
        all_voxel_centers: (K, 3) array of all occupied voxel centers.
    """
    rng = np.random.default_rng(seed)
    n_joints = len(joint_limits)
    limits = np.array(joint_limits)

    # Voxelize the task workspace
    mins = task_positions.min(axis=0) - grid_resolution
    maxs = task_positions.max(axis=0) + grid_resolution

    # Assign each point to a voxel
    voxel_indices = np.floor((task_positions - mins) / grid_resolution).astype(int)
    unique_voxels = np.unique(voxel_indices, axis=0)

    # Compute voxel centers
    all_voxel_centers = mins + (unique_voxels + 0.5) * grid_resolution

    print(f"  Voxelized task workspace: {len(unique_voxels)} occupied voxels "
          f"(resolution={grid_resolution:.3f} m)")

    # Set batch size: max 150000 samples to keep runtime under ~30s while maintaining accuracy
    n_batch = min(150000, max(50000, len(unique_voxels) * 8))
    q_batch = rng.uniform(limits[:, 0], limits[:, 1], size=(n_batch, n_joints))

    batch_pos = np.zeros((n_batch, 3))
    batch_yaw = np.zeros(n_batch)

    print(f"  Computing {n_batch} FK evaluations for dexterous analysis...")
    for i in range(n_batch):
        T = fk_func(q_batch[i])
        batch_pos[i] = T[:3, 3]
        # Extract yaw (rotation about world Z axis) from rotation matrix
        # yaw = atan2(R[1,0], R[0,0])
        R = T[:3, :3]
        batch_yaw[i] = np.arctan2(R[1, 0], R[0, 0])

    # Assign batch samples to voxels
    batch_voxel_idx = np.floor((batch_pos - mins) / grid_resolution).astype(int)

    # For each occupied voxel, check yaw coverage
    yaw_bins = np.linspace(-np.pi, np.pi, n_yaw_tests + 1)  # bin edges
    voxel_dict = {}  # map voxel tuple → set of yaw bins hit

    for i in range(n_batch):
        key = tuple(batch_voxel_idx[i])
        if key not in voxel_dict:
            voxel_dict[key] = set()
        yaw_bin = np.searchsorted(yaw_bins[1:], batch_yaw[i])
        yaw_bin = min(yaw_bin, n_yaw_tests - 1)
        voxel_dict[key].add(yaw_bin)

    # Filter: dexterous voxels must have coverage of most yaw bins
    dex_threshold = n_yaw_tests * 0.75  # 75% of yaw bins must be achievable
    dexterous_centers = []
    dex_scores = []
    for v in unique_voxels:
        key = tuple(v)
        center = mins + (v + 0.5) * grid_resolution
        if key in voxel_dict:
            coverage = len(voxel_dict[key]) / n_yaw_tests
            if len(voxel_dict[key]) >= dex_threshold:
                dexterous_centers.append(center)
                dex_scores.append(coverage)

    dexterous_centers = np.array(dexterous_centers) if dexterous_centers else np.empty((0, 3))
    dex_scores = np.array(dex_scores) if dex_scores else np.empty(0)

    return dexterous_centers, dex_scores, all_voxel_centers


# ═══════════════════════════════════════════════════════════════════
# 5. VISUALIZATION
# ═══════════════════════════════════════════════════════════════════

# Default table parameters from arena_description/table_arena.xml
# Table body at pos=[0,0,0.4] with box half-extents [0.6, 0.6, 0.4]
# → table top surface at z=0.8m, spanning [-0.6,0.6] in X and Y
TABLE_CENTER = [0.0, 0.0]   # x, y center of table in world frame
TABLE_SIZE   = [1.2, 1.2]   # width (x), depth (y) of table (full extents)
TABLE_HEIGHT = 0.8           # table surface height (z)


def plot_3d_workspace(task_pos, dex_pos, robot_name, output_dir, dex_scores=None):
    """
    Create 3D scatter plot showing task workspace (blue) and dexterous
    workspace (red) as point clouds.
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Task workspace (semi-transparent blue)
    subsample = min(len(task_pos), 20000)
    idx = np.random.choice(len(task_pos), subsample, replace=False)
    ax.scatter(task_pos[idx, 0], task_pos[idx, 1], task_pos[idx, 2],
               c='dodgerblue', alpha=0.08, s=1, label=f'Task Workspace ({len(task_pos)} pts)')

    # Dexterous workspace (red, more opaque)
    if len(dex_pos) > 0:
        if dex_scores is not None and len(dex_scores) > 0:
            colors = cm.hot(Normalize(vmin=0.75, vmax=1.0)(dex_scores))
            ax.scatter(dex_pos[:, 0], dex_pos[:, 1], dex_pos[:, 2],
                       c=colors, alpha=0.5, s=8,
                       label=f'Dexterous Workspace ({len(dex_pos)} voxels)')
        else:
            ax.scatter(dex_pos[:, 0], dex_pos[:, 1], dex_pos[:, 2],
                       c='red', alpha=0.5, s=8,
                       label=f'Dexterous Workspace ({len(dex_pos)} voxels)')

    # Draw table surface as a translucent rectangle (do not pass label to plot_surface)
    table_x = [TABLE_CENTER[0] - TABLE_SIZE[0]/2, TABLE_CENTER[0] + TABLE_SIZE[0]/2]
    table_y = [TABLE_CENTER[1] - TABLE_SIZE[1]/2, TABLE_CENTER[1] + TABLE_SIZE[1]/2]
    xx, yy = np.meshgrid(table_x, table_y)
    zz = np.full_like(xx, TABLE_HEIGHT)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='green')

    # Draw robot base
    ax.scatter([0], [0], [0], c='black', s=100, marker='^')

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(f'{robot_name} — Task vs Dexterous Workspace (3D)', fontsize=14, fontweight='bold')

    # Build custom legend handles to avoid matplotlib 3D legend bugs
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='dodgerblue',
               markersize=6, alpha=0.7, label=f'Task Workspace ({len(task_pos)} pts)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=6, alpha=0.7, label=f'Dexterous Workspace ({len(dex_pos)} voxels)'),
        Patch(facecolor='green', edgecolor='green', alpha=0.3, label='Table Surface'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='black',
               markersize=8, label='Robot Base')
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=9)

    # Set equal aspect ratio
    all_pts = task_pos if len(dex_pos) == 0 else np.vstack([task_pos, dex_pos])
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2.0
    mid = all_pts.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()
    fname = os.path.join(output_dir, f'{robot_name.lower().replace(" ", "_")}_workspace_3d.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()


def plot_2d_occupancy(task_pos, dex_pos, robot_name, output_dir,
                       grid_res=0.02, z_slice=None):
    """
    Create 2D top-down occupancy map (XY plane) with table region overlay.
    If z_slice is given, only show points within ±z_slice_width of that height.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Filter to table-height region if requested
    if z_slice is not None:
        z_width = 0.15  # ±15cm around table height
        mask_task = np.abs(task_pos[:, 2] - z_slice) < z_width
        task_xy = task_pos[mask_task, :2]
        if len(dex_pos) > 0:
            mask_dex = np.abs(dex_pos[:, 2] - z_slice) < z_width
            dex_xy = dex_pos[mask_dex, :2]
        else:
            dex_xy = np.empty((0, 2))
        slice_label = f' (z ≈ {z_slice:.2f}m ± {z_width:.2f}m)'
    else:
        task_xy = task_pos[:, :2]
        dex_xy = dex_pos[:, :2] if len(dex_pos) > 0 else np.empty((0, 2))
        slice_label = ' (all heights)'

    for ax_idx, (data, title, cmap_name) in enumerate([
        (task_xy, f'Task Workspace{slice_label}', 'Blues'),
        (dex_xy, f'Dexterous Workspace{slice_label}', 'Reds'),
    ]):
        ax = axes[ax_idx]

        if len(data) > 0:
            # Create 2D histogram / occupancy map
            x_range = [data[:, 0].min() - 0.1, data[:, 0].max() + 0.1]
            y_range = [data[:, 1].min() - 0.1, data[:, 1].max() + 0.1]
            bins_x = int((x_range[1] - x_range[0]) / grid_res)
            bins_y = int((y_range[1] - y_range[0]) / grid_res)
            H, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1],
                                                bins=[max(bins_x, 1), max(bins_y, 1)],
                                                range=[x_range, y_range])
            # Normalize and plot
            H_norm = H / H.max() if H.max() > 0 else H
            im = ax.imshow(H_norm.T, origin='lower', cmap=cmap_name,
                          extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                          aspect='equal', interpolation='bilinear')
            plt.colorbar(im, ax=ax, label='Relative Density', shrink=0.8)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)

        # Draw table region
        table_rect = Rectangle(
            (TABLE_CENTER[0] - TABLE_SIZE[0]/2, TABLE_CENTER[1] - TABLE_SIZE[1]/2),
            TABLE_SIZE[0], TABLE_SIZE[1],
            linewidth=2, edgecolor='green', facecolor='green',
            alpha=0.15, linestyle='--', label='Table Region'
        )
        ax.add_patch(table_rect)

        # Mark robot base
        ax.plot(0, 0, 'k^', markersize=12, label='Robot Base')

        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    fig.suptitle(f'{robot_name} — Occupancy Maps (Top-Down View)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = os.path.join(output_dir, f'{robot_name.lower().replace(" ", "_")}_workspace_2d.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()


def plot_feasible_table_regions(task_pos, dex_pos, robot_name, output_dir,
                                 table_height=0.0, z_tolerance=0.15,
                                 grid_res=0.015):
    """
    Highlight feasible table regions: show which parts of the table are
    reachable (task workspace) and dexterous.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Filter to table height
    mask_task = np.abs(task_pos[:, 2] - table_height) < z_tolerance
    task_table = task_pos[mask_task]

    if len(dex_pos) > 0:
        mask_dex = np.abs(dex_pos[:, 2] - table_height) < z_tolerance
        dex_table = dex_pos[mask_dex]
    else:
        dex_table = np.empty((0, 3))

    # Table boundaries
    tx_lo = TABLE_CENTER[0] - TABLE_SIZE[0]/2
    tx_hi = TABLE_CENTER[0] + TABLE_SIZE[0]/2
    ty_lo = TABLE_CENTER[1] - TABLE_SIZE[1]/2
    ty_hi = TABLE_CENTER[1] + TABLE_SIZE[1]/2

    # Create grid over table
    x_bins = np.arange(tx_lo, tx_hi + grid_res, grid_res)
    y_bins = np.arange(ty_lo, ty_hi + grid_res, grid_res)

    # Compute task workspace coverage on table
    task_on_table = task_table[
        (task_table[:, 0] >= tx_lo) & (task_table[:, 0] <= tx_hi) &
        (task_table[:, 1] >= ty_lo) & (task_table[:, 1] <= ty_hi)
    ] if len(task_table) > 0 else np.empty((0, 3))

    H_task, _, _ = np.histogram2d(
        task_on_table[:, 0] if len(task_on_table) > 0 else [],
        task_on_table[:, 1] if len(task_on_table) > 0 else [],
        bins=[x_bins, y_bins]
    ) if len(task_on_table) > 0 else (np.zeros((len(x_bins)-1, len(y_bins)-1)), x_bins, y_bins)

    # Compute dexterous coverage on table
    dex_on_table = dex_table[
        (dex_table[:, 0] >= tx_lo) & (dex_table[:, 0] <= tx_hi) &
        (dex_table[:, 1] >= ty_lo) & (dex_table[:, 1] <= ty_hi)
    ] if len(dex_table) > 0 else np.empty((0, 3))

    H_dex, _, _ = np.histogram2d(
        dex_on_table[:, 0] if len(dex_on_table) > 0 else [],
        dex_on_table[:, 1] if len(dex_on_table) > 0 else [],
        bins=[x_bins, y_bins]
    ) if len(dex_on_table) > 0 else (np.zeros((len(x_bins)-1, len(y_bins)-1)), x_bins, y_bins)

    # Create composite image: task=blue channel, dexterous=red channel
    task_norm = H_task / H_task.max() if H_task.max() > 0 else H_task
    dex_norm = H_dex / H_dex.max() if H_dex.max() > 0 else H_dex

    # RGB image: R=dexterous, G=overlap, B=task-only
    rgb = np.zeros((*task_norm.T.shape, 3))
    rgb[:, :, 2] = task_norm.T      # Blue = task workspace
    rgb[:, :, 0] = dex_norm.T       # Red = dexterous workspace
    rgb[:, :, 1] = np.minimum(task_norm.T, dex_norm.T) * 0.5  # slight green for overlap

    ax.imshow(rgb, origin='lower',
              extent=[tx_lo, tx_hi, ty_lo, ty_hi],
              aspect='equal', interpolation='bilinear')

    # Table border
    table_rect = Rectangle(
        (tx_lo, ty_lo), TABLE_SIZE[0], TABLE_SIZE[1],
        linewidth=2.5, edgecolor='limegreen', facecolor='none',
        linestyle='-', label='Table Border'
    )
    ax.add_patch(table_rect)

    # Robot base
    ax.plot(0, 0, 'w^', markersize=14, markeredgecolor='black', markeredgewidth=1.5,
            label='Robot Base')

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.6, label='Task Workspace (reachable)'),
        Patch(facecolor='red', alpha=0.6, label='Dexterous Workspace (yaw-flexible)'),
        Patch(facecolor='purple', alpha=0.6, label='Overlap (both)'),
        Patch(facecolor='none', edgecolor='limegreen', linewidth=2, label='Table Border'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
              facecolor='white', framealpha=0.8)

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(f'{robot_name} — Feasible Table Regions\n'
                 f'(z ≈ {table_height:.2f}m ± {z_tolerance:.2f}m)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2, color='white')

    # Stats annotation
    n_task_cells = np.sum(H_task > 0)
    n_dex_cells = np.sum(H_dex > 0)
    n_total_cells = max((len(x_bins)-1) * (len(y_bins)-1), 1)
    stats_text = (f'Table Coverage:\n'
                  f'  Task: {n_task_cells}/{n_total_cells} cells '
                  f'({100*n_task_cells/n_total_cells:.1f}%)\n'
                  f'  Dexterous: {n_dex_cells}/{n_total_cells} cells '
                  f'({100*n_dex_cells/n_total_cells:.1f}%)')
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    fname = os.path.join(output_dir, f'{robot_name.lower().replace(" ", "_")}_table_feasibility.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fname}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════

def run_analysis(robot_name, fk_func, joint_limits, num_samples, output_dir):
    """Run complete workspace analysis for one robot."""
    print(f"\n{'='*65}")
    print(f"  WORKSPACE ANALYSIS: {robot_name}")
    print(f"{'='*65}")
    n_joints = len(joint_limits)
    print(f"  DOF: {n_joints}")
    print(f"  Joint Limits:")
    for i, (lo, hi) in enumerate(joint_limits):
        print(f"    J{i+1}: [{lo:+.4f}, {hi:+.4f}] rad  "
              f"([{np.degrees(lo):+.1f}°, {np.degrees(hi):+.1f}°])")

    # --- Task Workspace ---
    print(f"\n  [1/3] Computing Task Workspace ({num_samples} random samples)...")
    t0 = time.time()
    task_pos, task_ori = compute_task_workspace(fk_func, joint_limits, num_samples)
    t1 = time.time()
    print(f"        Done in {t1-t0:.1f}s")
    print(f"        Position range:")
    print(f"          X: [{task_pos[:,0].min():.3f}, {task_pos[:,0].max():.3f}] m")
    print(f"          Y: [{task_pos[:,1].min():.3f}, {task_pos[:,1].max():.3f}] m")
    print(f"          Z: [{task_pos[:,2].min():.3f}, {task_pos[:,2].max():.3f}] m")

    # --- Dexterous Workspace ---
    print(f"\n  [2/3] Computing Dexterous Workspace...")
    t0 = time.time()
    dex_pos, dex_scores, voxel_centers = compute_dexterous_workspace(
        fk_func, joint_limits, task_pos,
        grid_resolution=0.04, n_yaw_tests=8
    )
    t1 = time.time()
    print(f"        Done in {t1-t0:.1f}s")
    print(f"        Dexterous voxels: {len(dex_pos)} / {len(voxel_centers)} "
          f"({100*len(dex_pos)/max(len(voxel_centers),1):.1f}%)")

    # --- Visualization ---
    print(f"\n  [3/3] Generating visualizations...")
    plot_3d_workspace(task_pos, dex_pos, robot_name, output_dir, dex_scores)
    plot_2d_occupancy(task_pos, dex_pos, robot_name, output_dir,
                      z_slice=TABLE_HEIGHT, grid_res=0.02)
    plot_feasible_table_regions(task_pos, dex_pos, robot_name, output_dir,
                                table_height=TABLE_HEIGHT)

    print(f"\n  ✅ {robot_name} analysis complete!")
    return task_pos, dex_pos


def main():
    parser = argparse.ArgumentParser(
        description='Task vs Dexterous Workspace Analysis for Robot Manipulators'
    )
    parser.add_argument('--robot', type=str, default='both',
                        choices=['franka', 'heal', 'both'],
                        help='Which robot to analyze (default: both)')
    parser.add_argument('--samples', type=int, default=50000,
                        help='Number of Monte Carlo samples (default: 50000)')
    parser.add_argument('--output', type=str, default='workspace_results',
                        help='Output directory for plots (default: workspace_results)')
    parser.add_argument('--table-x', type=float, default=0.0,
                        help='Table center X position (default: 0.0)')
    parser.add_argument('--table-y', type=float, default=0.0,
                        help='Table center Y position (default: 0.0)')
    parser.add_argument('--table-height', type=float, default=0.8,
                        help='Table surface height (default: 0.8)')
    parser.add_argument('--table-width', type=float, default=1.2,
                        help='Table width in X (default: 1.2)')
    parser.add_argument('--table-depth', type=float, default=1.2,
                        help='Table depth in Y (default: 1.2)')

    args = parser.parse_args()

    # Update table parameters
    global TABLE_CENTER, TABLE_SIZE, TABLE_HEIGHT
    TABLE_CENTER = [args.table_x, args.table_y]
    TABLE_SIZE = [args.table_width, args.table_depth]
    TABLE_HEIGHT = args.table_height

    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    if args.robot in ('franka', 'both'):
        run_analysis('Franka Panda', analytical_fk_franka, FRANKA_JOINT_LIMITS,
                     args.samples, output_dir)

    if args.robot in ('heal', 'both'):
        run_analysis('Addverb HEAL', analytical_fk_heal, HEAL_JOINT_LIMITS,
                     args.samples, output_dir)

    print(f"\n{'='*65}")
    print(f"  ALL ANALYSES COMPLETE — results in: {output_dir}")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()
