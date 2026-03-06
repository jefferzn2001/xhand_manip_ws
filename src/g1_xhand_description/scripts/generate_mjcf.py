#!/usr/bin/env python3
"""
Generate G1+XHand MJCF by modifying the stock G1 MJCF directly.

Instead of compiling from URDF (which loses visual meshes, sensors, contacts),
this script takes the working stock G1 MJCF and replaces the rubber hands
with XHand body trees parsed from the XHand URDFs.

Preserves: visual meshes, IMU sensors, contact pairs, tendons, defaults,
floor, lighting, skybox -- everything that makes the stock G1 work.

Usage:
    python generate_mjcf.py [--stock-mjcf <path>] [--output <path>]
"""

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

HAND_MASS_TARGET = 1.169  # kg, user-measured per hand
HAND_OFFSET_RIGHT = "0.0465 -0.003 0"  # stock 0.0415 + 5 mm gap
HAND_OFFSET_LEFT = "0.0465 0.003 0"
# Reason: XHand fingers extend along +Z but G1 arm extends along +X.
# Pitch +90° aligns Z→X (fingers forward), then local yaw ±90° twists
# the hand so thumbs point up and palms face inward (natural rest pose).
HAND_ROTATION_QUAT_RIGHT = "0.5 0.5 0.5 0.5"
HAND_ROTATION_QUAT_LEFT = "0.5 -0.5 0.5 -0.5"
MIN_MASS = 1e-4
MIN_INERTIA = 1e-8


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> tuple:
    """
    Convert RPY (XYZ extrinsic) angles to MuJoCo wxyz quaternion.

    Args:
        roll: Roll angle in radians.
        pitch: Pitch angle in radians.
        yaw: Yaw angle in radians.

    Returns:
        tuple: (w, x, y, z) quaternion.
    """
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _fmt_quat(q: tuple) -> str:
    """Format quaternion tuple as MuJoCo attribute string."""
    return f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"


def parse_xhand_urdf(urdf_path: str) -> tuple:
    """
    Parse XHand URDF into link data, kinematic tree, and root link name.

    Args:
        urdf_path: Path to XHand URDF file.

    Returns:
        tuple: (links dict, children_of dict, root_link_name)
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links = {}
    for link_elem in root.findall("link"):
        name = link_elem.get("name")
        info = {"name": name, "mass": 0.0, "inertial_pos": "0 0 0"}

        inertial = link_elem.find("inertial")
        if inertial is not None:
            mass_e = inertial.find("mass")
            if mass_e is not None:
                info["mass"] = float(mass_e.get("value", "0"))
            origin_e = inertial.find("origin")
            if origin_e is not None:
                info["inertial_pos"] = origin_e.get("xyz", "0 0 0")
            inertia_e = inertial.find("inertia")
            if inertia_e is not None:
                info["inertia"] = {
                    k: float(inertia_e.get(k, "0"))
                    for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
                }

        visual = link_elem.find("visual")
        if visual is not None:
            mesh = visual.find("geometry/mesh")
            if mesh is not None:
                info["mesh_file"] = mesh.get("filename", "").split("/")[-1]

        links[name] = info

    children_of = {}
    child_links = set()
    for joint_elem in root.findall("joint"):
        jname = joint_elem.get("name")
        jtype = joint_elem.get("type")
        parent = joint_elem.find("parent").get("link")
        child = joint_elem.find("child").get("link")

        origin = joint_elem.find("origin")
        xyz = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
        rpy = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"

        axis_e = joint_elem.find("axis")
        axis = axis_e.get("xyz", "0 0 1") if axis_e is not None else "0 0 1"

        limit = joint_elem.find("limit")
        lower = float(limit.get("lower", "0")) if limit is not None else 0
        upper = float(limit.get("upper", "0")) if limit is not None else 0

        jinfo = {
            "name": jname,
            "type": jtype,
            "xyz": xyz,
            "rpy": rpy,
            "axis": axis,
            "lower": lower,
            "upper": upper,
        }
        children_of.setdefault(parent, []).append((jinfo, child))
        child_links.add(child)

    root_link = next(
        (name for name in links if name not in child_links), None
    )
    return links, children_of, root_link


def build_xhand_body(
    root_link: str, links: dict, children_of: dict, scale: float
) -> ET.Element:
    """
    Convert XHand URDF kinematic tree into a MuJoCo <body> XML subtree.

    Fixed-joint children are merged into the parent body (visual geoms only).
    Revolute-joint children become nested <body> elements with <joint>.

    Args:
        root_link: Name of the root hand link.
        links: Dict of link data from parse_xhand_urdf.
        children_of: Parent -> [(joint_info, child_name)] mapping.
        scale: Mass scale factor to match measured weight.

    Returns:
        ET.Element: Root <body> element for the hand.
    """

    def _make_body(link_name: str) -> ET.Element:
        link = links[link_name]
        body = ET.Element("body")
        body.set("name", link_name)

        mass = link.get("mass", 0) * scale
        inertial = ET.SubElement(body, "inertial")
        inertial.set("pos", link.get("inertial_pos", "0 0 0"))
        if mass >= MIN_MASS:
            inertial.set("mass", f"{mass:.6f}")
            ii = link.get("inertia")
            if ii:
                ixx = max(ii["ixx"] * scale, MIN_INERTIA)
                iyy = max(ii["iyy"] * scale, MIN_INERTIA)
                izz = max(ii["izz"] * scale, MIN_INERTIA)
                ixy, ixz, iyz = (
                    ii["ixy"] * scale,
                    ii["ixz"] * scale,
                    ii["iyz"] * scale,
                )
                inertial.set(
                    "fullinertia",
                    f"{ixx:.8e} {iyy:.8e} {izz:.8e} "
                    f"{ixy:.8e} {ixz:.8e} {iyz:.8e}",
                )
            else:
                inertial.set(
                    "diaginertia",
                    f"{MIN_INERTIA} {MIN_INERTIA} {MIN_INERTIA}",
                )
        else:
            inertial.set("mass", str(MIN_MASS))
            inertial.set(
                "diaginertia",
                f"{MIN_INERTIA} {MIN_INERTIA} {MIN_INERTIA}",
            )

        if "mesh_file" in link:
            geom = ET.SubElement(body, "geom")
            geom.set("type", "mesh")
            geom.set("mesh", Path(link["mesh_file"]).stem)
            geom.set("class", "visual")

        for jinfo, child_name in children_of.get(link_name, []):
            if jinfo["type"] == "fixed":
                child_link = links.get(child_name, {})
                if "mesh_file" in child_link:
                    g = ET.SubElement(body, "geom")
                    g.set("type", "mesh")
                    g.set("mesh", Path(child_link["mesh_file"]).stem)
                    g.set("class", "visual")
                    g.set("pos", jinfo["xyz"])
                    rpy_vals = [float(v) for v in jinfo["rpy"].split()]
                    if any(abs(v) > 1e-10 for v in rpy_vals):
                        g.set("quat", _fmt_quat(rpy_to_quat(*rpy_vals)))
            else:
                child_body = _make_body(child_name)
                child_body.set("pos", jinfo["xyz"])
                rpy_vals = [float(v) for v in jinfo["rpy"].split()]
                if any(abs(v) > 1e-10 for v in rpy_vals):
                    child_body.set(
                        "quat", _fmt_quat(rpy_to_quat(*rpy_vals))
                    )
                joint_elem = ET.Element("joint")
                joint_elem.set("name", jinfo["name"])
                joint_elem.set("axis", jinfo["axis"])
                joint_elem.set("range", f"{jinfo['lower']} {jinfo['upper']}")
                joint_elem.set("class", "xhand")
                # Reason: insert after inertial so MuJoCo sees joint before geoms
                child_body.insert(1, joint_elem)
                body.append(child_body)

        return body

    return _make_body(root_link)


def main():
    """Entry point for MJCF generation."""
    parser = argparse.ArgumentParser(
        description="Generate G1+XHand MJCF by modifying stock G1 MJCF"
    )
    parser.add_argument(
        "--stock-mjcf",
        default="/opt/ros/humble/share/unitree_description/mjcf/g1.xml",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--xhand-right", default=None)
    parser.add_argument("--xhand-left", default=None)
    args = parser.parse_args()

    pkg_dir = Path(__file__).resolve().parent.parent
    output = args.output or str(pkg_dir / "mjcf" / "g1_xhand.xml")
    xhand_r = args.xhand_right or str(
        pkg_dir / "urdf" / "xhand_source" / "xhand_right.urdf"
    )
    xhand_l = args.xhand_left or str(
        pkg_dir / "urdf" / "xhand_source" / "xhand_left.urdf"
    )

    Path(output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Stock MJCF:  {args.stock_mjcf}")
    print(f"XHand Right: {xhand_r}")
    print(f"XHand Left:  {xhand_l}")
    print(f"Output:      {output}")
    print()

    # --- Step 1: Parse inputs ---
    print("Step 1: Parsing stock G1 MJCF and XHand URDFs...")
    stock_text = Path(args.stock_mjcf).read_text()
    stock_text = stock_text.replace("<contact>>", "<contact>")
    mjcf = ET.fromstring(stock_text)

    r_links, r_children, r_root = parse_xhand_urdf(xhand_r)
    l_links, l_children, l_root = parse_xhand_urdf(xhand_l)

    r_total = sum(lk["mass"] for lk in r_links.values())
    l_total = sum(lk["mass"] for lk in l_links.values())
    r_scale = HAND_MASS_TARGET / r_total if r_total > 0 else 1.0
    l_scale = HAND_MASS_TARGET / l_total if l_total > 0 else 1.0
    print(
        f"  Right hand: {r_total:.4f} kg -> {HAND_MASS_TARGET} kg "
        f"(scale {r_scale:.4f})"
    )
    print(
        f"  Left hand:  {l_total:.4f} kg -> {HAND_MASS_TARGET} kg "
        f"(scale {l_scale:.4f})"
    )

    # --- Step 2: Update model name and mesh paths ---
    print("Step 2: Updating mesh paths...")
    mjcf.set("model", "g1_xhand")

    compiler = mjcf.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", "../meshes")

    asset = mjcf.find("asset")
    rubber_to_remove = []
    for mesh in asset.findall("mesh"):
        name = mesh.get("name", "")
        if name in ("left_rubber_hand", "right_rubber_hand"):
            rubber_to_remove.append(mesh)
        else:
            old_file = mesh.get("file", "")
            mesh.set("file", f"g1/{old_file}")

    for m in rubber_to_remove:
        asset.remove(m)
        print(f"  Removed rubber hand mesh: {m.get('name')}")

    xhand_meshes = set()
    for links_dict in (r_links, l_links):
        for lk in links_dict.values():
            if "mesh_file" in lk:
                xhand_meshes.add(lk["mesh_file"])

    for mf in sorted(xhand_meshes):
        m = ET.SubElement(asset, "mesh")
        m.set("name", Path(mf).stem)
        m.set("file", mf)
    print(f"  Added {len(xhand_meshes)} XHand mesh assets")

    # --- Step 3: Add XHand joint default class ---
    print("Step 3: Adding XHand joint defaults...")
    defaults = mjcf.find("default")
    if defaults is not None:
        g1_default = defaults.find("default[@class='g1']")
        parent = g1_default if g1_default is not None else defaults
        xhand_def = ET.SubElement(parent, "default")
        xhand_def.set("class", "xhand")
        xjoint = ET.SubElement(xhand_def, "joint")
        xjoint.set("frictionloss", "0.02")
        xjoint.set("armature", "0.001")
        xjoint.set("damping", "0.05")

    # --- Step 4: Replace rubber hands with XHand bodies ---
    print("Step 4: Replacing rubber hands with XHand bodies...")
    worldbody = mjcf.find("worldbody")

    configs = [
        ("right", r_links, r_children, r_root, r_scale,
         HAND_OFFSET_RIGHT, HAND_ROTATION_QUAT_RIGHT),
        ("left", l_links, l_children, l_root, l_scale,
         HAND_OFFSET_LEFT, HAND_ROTATION_QUAT_LEFT),
    ]

    for side, links, children, root_link, scale, offset, rot_quat in configs:
        wrist = worldbody.find(f".//body[@name='{side}_wrist_yaw_link']")
        if wrist is None:
            print(f"  WARNING: {side}_wrist_yaw_link not found!")
            continue

        for geom in list(wrist.findall("geom")):
            if "rubber_hand" in geom.get("mesh", ""):
                wrist.remove(geom)
            elif "hand_collision" in geom.get("name", ""):
                wrist.remove(geom)

        for site in list(wrist.findall("site")):
            if "palm" in site.get("name", ""):
                wrist.remove(site)

        xhand_body = build_xhand_body(root_link, links, children, scale)
        xhand_body.set("pos", offset)
        xhand_body.set("quat", rot_quat)

        palm = ET.SubElement(xhand_body, "site")
        palm.set("name", f"{side}_palm")
        palm.set("pos", "0 0 0.08")
        palm.set("size", "0.01")

        wrist.append(xhand_body)
        print(f"  Inserted {side} XHand body tree")

    # --- Step 5: Write output ---
    print(f"\nStep 5: Writing MJCF to {output}")
    ET.indent(mjcf, space="    ")
    tree = ET.ElementTree(mjcf)
    tree.write(output, encoding="unicode", xml_declaration=True)

    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(output)
        print(
            f"  MuJoCo verification OK: {model.nbody} bodies, "
            f"{model.njnt} joints, {model.nsensor} sensors"
        )
    except ImportError:
        print("  (mujoco not installed, skipping verification)")
    except Exception as e:
        print(f"  MuJoCo verification FAILED: {e}")

    print(f"\nDone! MJCF written to: {output}")


if __name__ == "__main__":
    main()
