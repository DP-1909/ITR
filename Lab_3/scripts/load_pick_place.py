"""Load the workspace-sized HEAL pick-and-place scene in the MuJoCo viewer."""

import argparse
from pathlib import Path

import mujoco
from mujoco.viewer import launch_passive

HERE = Path(__file__).resolve().parent
LAB = HERE.parent


def resolve_scene_xml(requested: str | None = None) -> Path:
    """Resolve an explicit XML path or find the generated scene from this file."""
    if requested:
        xml = Path(requested).expanduser()
        if not xml.is_absolute():
            cwd_candidate = xml.resolve()
            lab_candidate = (LAB / xml).resolve()
            if cwd_candidate.is_file():
                return cwd_candidate
            if lab_candidate.is_file():
                return lab_candidate
            raise FileNotFoundError(
                "Scene XML not found. Searched:\n"
                f"  - {cwd_candidate}\n"
                f"  - {lab_candidate}"
            )

        xml = xml.resolve()
        if xml.is_file():
            return xml
        raise FileNotFoundError(f"Scene XML not found: {xml}")

    candidates = (
        LAB / "robot_descriptions" / "heal_pick_place.xml",
        LAB / "heal_pick_place.xml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "Pick-and-place scene XML is missing.\n"
        f"{searched}\n"
        f"Generate it with: python3 {LAB / 'scripts' / 'table_design.py'}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "xml",
        nargs="?",
        help="Optional path to a scene XML (absolute or relative to the lab directory)",
    )
    args = parser.parse_args()

    xml = resolve_scene_xml(args.xml)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    # Hold the arm against gravity; the cube is a free joint and should settle on the table.
    arm_acts = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        for n in ("turret", "shoulder", "elbow", "wrist_1", "wrist_2", "wrist_3")
    ]
    mujoco.mj_forward(model, data)
    with launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            for aid in arm_acts:
                if aid < 0:
                    continue
                jnt_dof = model.actuator_trnid[aid][0]
                data.ctrl[aid] = data.qfrc_bias[jnt_dof]
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
