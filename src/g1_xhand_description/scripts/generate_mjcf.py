#!/usr/bin/env python3
"""
Generate G1+XHand MJCF for MuJoCo simulation.

Converts the combined URDF to MJCF using MuJoCo's compiler, then merges
in physics properties (contact, tendons, sensors) from the stock G1 MJCF.

Usage:
    python generate_mjcf.py [--urdf <path>] [--output <path>]
"""

import argparse
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def resolve_mesh_paths(urdf_path: str, output_urdf_path: str, mesh_dir: str) -> None:
    """
    Rewrite package:// mesh paths in URDF to absolute paths for MuJoCo.
    Also fixes degenerate inertias that MuJoCo's compiler rejects.

    Args:
        urdf_path: Input URDF path.
        output_urdf_path: Output URDF path with resolved paths.
        mesh_dir: Absolute path to directory containing all mesh files.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for mesh in root.iter("mesh"):
        filename = mesh.get("filename", "")
        if "package://" in filename:
            mesh_file = filename.split("/meshes/")[-1]
            mesh.set("filename", str(Path(mesh_dir) / mesh_file))

    # Fix degenerate inertias -- XHand has dummy links (ee, back, tips,
    # rotaback) with near-zero mass and 1e-11 inertia values which
    # MuJoCo rejects ("inertia must have positive eigenvalues").
    MIN_INERTIA = 1e-8
    MIN_MASS = 1e-4
    fixed_count = 0
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass_elem = inertial.find("mass")
        inertia_elem = inertial.find("inertia")
        if mass_elem is None or inertia_elem is None:
            continue

        mass_val = float(mass_elem.get("value", "0"))
        if mass_val < MIN_MASS:
            mass_elem.set("value", str(MIN_MASS))
            for attr in ["ixx", "iyy", "izz"]:
                val = float(inertia_elem.get(attr, "0"))
                if val < MIN_INERTIA:
                    inertia_elem.set(attr, str(MIN_INERTIA))
            for attr in ["ixy", "ixz", "iyz"]:
                inertia_elem.set(attr, "0")
            fixed_count += 1

    if fixed_count > 0:
        print(f"  Fixed {fixed_count} degenerate inertias for MuJoCo compatibility")

    tree.write(output_urdf_path, encoding="utf-8", xml_declaration=True)


def compile_urdf_to_mjcf(urdf_path: str, mjcf_path: str) -> None:
    """
    Compile URDF to MJCF using MuJoCo's compiler.

    Args:
        urdf_path: Path to URDF file.
        mjcf_path: Path to write MJCF output.
    """
    try:
        import mujoco
    except ImportError:
        print("ERROR: mujoco not installed. Install with: pip install mujoco")
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(urdf_path)
    mujoco.mj_saveLastXML(mjcf_path, model)
    print(f"  Compiled URDF to MJCF: {mjcf_path}")
    print(f"  Bodies: {model.nbody}, Joints: {model.njnt}, Actuators: {model.nu}")


def merge_physics_from_stock(
    compiled_mjcf: str, stock_mjcf: str, output_mjcf: str, mesh_dir: str
) -> None:
    """
    Merge physics properties from stock G1 MJCF into compiled G1+XHand MJCF.

    Copies: contact pairs, tendons, sensors, visual settings, defaults.

    Args:
        compiled_mjcf: Path to MuJoCo-compiled MJCF (from URDF).
        stock_mjcf: Path to stock G1 MJCF with physics properties.
        output_mjcf: Path to write final MJCF.
        mesh_dir: Absolute path to mesh directory for compiler meshdir.
    """
    compiled_tree = ET.parse(compiled_mjcf)
    compiled_root = compiled_tree.getroot()
    compiled_root.set("model", "g1_xhand")

    stock_tree = ET.parse(stock_mjcf)
    stock_root = stock_tree.getroot()

    # Update compiler meshdir to point to our mesh directory
    compiler = compiled_root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", mesh_dir)
    else:
        compiler = ET.SubElement(compiled_root, "compiler")
        compiler.set("angle", "radian")
        compiler.set("meshdir", mesh_dir)

    # Collect all geom and joint names in the compiled model for validation
    compiled_geoms = {
        g.get("name") for g in compiled_root.iter("geom") if g.get("name")
    }
    compiled_joints = {
        j.get("name") for j in compiled_root.iter("joint") if j.get("name")
    }

    # Copy contact section, filtering out pairs that reference missing geoms
    stock_contact = stock_root.find("contact")
    if stock_contact is not None and compiled_root.find("contact") is None:
        new_contact = ET.SubElement(compiled_root, "contact")
        kept = 0
        for pair in stock_contact.findall("pair"):
            g1 = pair.get("geom1", "")
            g2 = pair.get("geom2", "")
            # Keep only if both geoms exist (geom2='floor' added later)
            if g1 in compiled_geoms and (g2 in compiled_geoms or g2 == "floor"):
                new_contact.append(pair)
                kept += 1
        if kept == 0:
            compiled_root.remove(new_contact)
        else:
            print(f"  Kept {kept} contact pairs from stock MJCF")

    # Copy tendon section, filtering out entries that reference missing joints
    stock_tendon = stock_root.find("tendon")
    if stock_tendon is not None and compiled_root.find("tendon") is None:
        new_tendon = ET.SubElement(compiled_root, "tendon")
        kept = 0
        for fixed in stock_tendon.findall("fixed"):
            valid = True
            for jref in fixed.findall("joint"):
                if jref.get("joint", "") not in compiled_joints:
                    valid = False
                    break
            if valid:
                new_tendon.append(fixed)
                kept += 1
        if kept == 0:
            compiled_root.remove(new_tendon)
        else:
            print(f"  Kept {kept} tendons from stock MJCF")

    # Copy sensor section, filtering out sensors referencing missing sites
    compiled_sites = {
        s.get("name") for s in compiled_root.iter("site") if s.get("name")
    }
    stock_sensor = stock_root.find("sensor")
    if stock_sensor is not None and compiled_root.find("sensor") is None:
        new_sensor = ET.SubElement(compiled_root, "sensor")
        kept = 0
        for sensor_elem in stock_sensor:
            obj = sensor_elem.get("objname", "") or sensor_elem.get("site", "")
            if obj in compiled_sites or not obj:
                new_sensor.append(sensor_elem)
                kept += 1
        if kept == 0:
            compiled_root.remove(new_sensor)
        else:
            print(f"  Kept {kept} sensors from stock MJCF")

    # Copy visual settings from stock
    stock_visual = stock_root.find("visual")
    if stock_visual is not None:
        compiled_visual = compiled_root.find("visual")
        if compiled_visual is not None:
            compiled_root.remove(compiled_visual)
        compiled_root.append(stock_visual)

    # Add floor and lighting from stock (second worldbody)
    # Stock has two worldbody elements: one for robot, one for floor/lights
    stock_worldbodies = stock_root.findall("worldbody")
    if len(stock_worldbodies) > 1:
        floor_worldbody = stock_worldbodies[1]
        compiled_root.append(floor_worldbody)

    # Add skybox and ground textures from stock
    stock_assets = stock_root.findall("asset")
    if len(stock_assets) > 1:
        scene_asset = stock_assets[1]
        compiled_root.append(scene_asset)

    # Add default classes for finger joints
    defaults = compiled_root.find("default")
    if defaults is None:
        defaults = ET.SubElement(compiled_root, "default")

    xhand_default = ET.SubElement(defaults, "default")
    xhand_default.set("class", "xhand")
    xhand_joint = ET.SubElement(xhand_default, "joint")
    xhand_joint.set("frictionloss", "0.05")
    xhand_joint.set("solimplimit", "0.97 0.995 0.001")

    # Add IMU site if not present
    compiled_worldbody = compiled_root.find("worldbody")
    if compiled_worldbody is not None:
        pelvis = compiled_worldbody.find(".//body[@name='pelvis']")
        if pelvis is not None:
            # Check if imu site exists
            has_imu = any(
                s.get("name") == "imu_in_pelvis"
                for s in pelvis.findall("site")
            )
            if not has_imu:
                imu_site = ET.SubElement(pelvis, "site")
                imu_site.set("name", "imu_in_pelvis")
                imu_site.set("size", "0.01")
                imu_site.set("pos", "0.04525 0 -0.08339")

    compiled_tree.write(output_mjcf, encoding="utf-8", xml_declaration=True)
    print(f"  Merged physics from stock MJCF")
    print(f"  Output: {output_mjcf}")


def main():
    """Entry point for MJCF generation."""
    parser = argparse.ArgumentParser(
        description="Generate G1+XHand MJCF for MuJoCo simulation"
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default=None,
        help="Path to combined G1+XHand URDF",
    )
    parser.add_argument(
        "--stock-mjcf",
        type=str,
        default="/opt/ros/humble/share/unitree_description/mjcf/g1.xml",
        help="Path to stock G1 MJCF (for physics properties)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write final MJCF",
    )
    parser.add_argument(
        "--mesh-dir",
        type=str,
        default=None,
        help="Path to mesh directory",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    pkg_dir = script_dir.parent

    urdf_path = args.urdf or str(pkg_dir / "urdf" / "g1_xhand" / "main.urdf")
    output_path = args.output or str(pkg_dir / "mjcf" / "g1_xhand.xml")
    mesh_dir = args.mesh_dir or str(pkg_dir / "meshes")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Input URDF:  {urdf_path}")
    print(f"Stock MJCF:  {args.stock_mjcf}")
    print(f"Mesh dir:    {mesh_dir}")
    print(f"Output MJCF: {output_path}")
    print()

    # Step 1: Resolve mesh paths in URDF
    # Write resolved URDF next to meshes so MuJoCo can find them
    resolved_urdf = str(Path(mesh_dir) / "_resolved_g1_xhand.urdf")
    resolve_mesh_paths(urdf_path, resolved_urdf, mesh_dir)
    print("Step 1: Resolved mesh paths in URDF")

    # Step 2: Compile URDF to MJCF using MuJoCo
    compiled_mjcf = str(Path(mesh_dir) / "_compiled_g1_xhand.xml")
    print("Step 2: Compiling URDF to MJCF via MuJoCo...")
    compile_urdf_to_mjcf(resolved_urdf, compiled_mjcf)

    # Step 3: Merge physics from stock G1 MJCF
    print("Step 3: Merging physics from stock G1 MJCF...")
    merge_physics_from_stock(compiled_mjcf, args.stock_mjcf, output_path, mesh_dir)

    # Cleanup
    Path(resolved_urdf).unlink(missing_ok=True)
    Path(compiled_mjcf).unlink(missing_ok=True)

    print(f"\nDone! MJCF written to: {output_path}")


if __name__ == "__main__":
    main()
