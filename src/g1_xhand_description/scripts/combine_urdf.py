#!/usr/bin/env python3
"""
Combine Unitree G1 URDF with XHand dexterous hand URDFs.

Reads the G1 main.urdf, removes rubber hand links/joints,
and inserts XHand links/joints attached at the wrist.

Usage:
    python combine_urdf.py --g1-urdf <path> --xhand-right <path> --xhand-left <path> --output <path>
    python combine_urdf.py  # uses defaults from installed ROS2 packages
"""

import argparse
import copy
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional


# Attachment config matching xhand1/config.yaml
G1_REMOVE_LINKS = ["left_rubber_hand", "right_rubber_hand"]
G1_REMOVE_JOINTS = ["left_hand_palm_joint", "right_hand_palm_joint"]

# G1 wrist -> xhand attachment origins
# Stock G1 palm joint is at 41.5mm from wrist along X.
# Adding 5mm spacer between wrist end-effector and XHand palm = 46.5mm total.
# Reason: XHand fingers extend along +Z but G1 arm extends along +X,
# so we rotate +90° about Y (pitch) to align Z→X.
RIGHT_HAND_ORIGIN = {"xyz": "0.0465 -0.003 0", "rpy": "0 1.5708 0"}
LEFT_HAND_ORIGIN = {"xyz": "0.0465 0.003 0", "rpy": "0 1.5708 0"}


def parse_urdf(path: str) -> ET.ElementTree:
    """
    Parse a URDF XML file.

    Args:
        path: Path to the URDF file.

    Returns:
        ET.ElementTree: Parsed XML tree.
    """
    return ET.parse(path)


def remove_elements(root: ET.Element, links: list, joints: list) -> None:
    """
    Remove specified links and joints from a URDF root element.

    Args:
        root: The robot XML root element.
        links: List of link names to remove.
        joints: List of joint names to remove.
    """
    for link_name in links:
        for link_elem in root.findall("link"):
            if link_elem.get("name") == link_name:
                root.remove(link_elem)
                break

    for joint_name in joints:
        for joint_elem in root.findall("joint"):
            if joint_elem.get("name") == joint_name:
                root.remove(joint_elem)
                break


def extract_hand_elements(
    hand_tree: ET.ElementTree,
    mesh_package_name: str,
) -> list:
    """
    Extract all link and joint elements from a hand URDF.

    Args:
        hand_tree: Parsed XHand URDF tree.
        mesh_package_name: ROS package name for mesh references
                           (e.g. 'g1_xhand_description').

    Returns:
        list: List of XML elements (links and joints) to insert.
    """
    hand_root = hand_tree.getroot()
    elements = []

    for child in hand_root:
        if child.tag in ("link", "joint"):
            elem = copy.deepcopy(child)
            # Rewrite mesh paths: package://xhand_right/meshes/...
            # -> package://g1_xhand_description/meshes/...
            for mesh in elem.iter("mesh"):
                filename = mesh.get("filename", "")
                if "package://" in filename:
                    # Extract just the mesh filename
                    mesh_file = filename.split("/meshes/")[-1]
                    mesh.set(
                        "filename",
                        f"package://{mesh_package_name}/meshes/{mesh_file}",
                    )
            elements.append(elem)

    return elements


def combine_urdf(
    g1_urdf_path: str,
    xhand_right_path: str,
    xhand_left_path: str,
    output_path: str,
    mesh_package: str = "g1_xhand_description",
    measured_hand_mass: Optional[float] = None,
) -> None:
    """
    Combine G1 URDF with XHand URDFs.

    Steps:
        1. Parse G1 URDF
        2. Remove rubber hand links and joints
        3. Parse XHand URDFs
        4. Add fixed joints attaching XHand root links to G1 wrist links
        5. Insert all XHand links and joints
        6. Optionally scale hand mass to match measured weight

    Args:
        g1_urdf_path: Path to G1 main.urdf.
        xhand_right_path: Path to xhand_right.urdf.
        xhand_left_path: Path to xhand_left.urdf.
        output_path: Path to write combined URDF.
        mesh_package: ROS package name for mesh file references.
        measured_hand_mass: If provided, scale hand link masses so total
                           matches this value (kg). Per hand.
    """
    g1_tree = parse_urdf(g1_urdf_path)
    g1_root = g1_tree.getroot()

    # Step 1: Remove rubber hands
    remove_elements(g1_root, G1_REMOVE_LINKS, G1_REMOVE_JOINTS)

    # Step 2: Parse XHand URDFs
    right_tree = parse_urdf(xhand_right_path)
    left_tree = parse_urdf(xhand_left_path)

    right_elements = extract_hand_elements(right_tree, mesh_package)
    left_elements = extract_hand_elements(left_tree, mesh_package)

    # Step 3: Optionally scale masses
    if measured_hand_mass is not None:
        _scale_hand_masses(right_elements, measured_hand_mass)
        _scale_hand_masses(left_elements, measured_hand_mass)

    # Step 4: Create fixed joints attaching XHand to G1 wrists
    right_attach = _create_attachment_joint(
        name="right_hand_palm_joint",
        parent="right_wrist_yaw_link",
        child="right_hand_link",
        origin=RIGHT_HAND_ORIGIN,
    )
    left_attach = _create_attachment_joint(
        name="left_hand_palm_joint",
        parent="left_wrist_yaw_link",
        child="left_hand_link",
        origin=LEFT_HAND_ORIGIN,
    )

    # Step 5: Insert into G1 URDF
    # Find insertion point (after the wrist links, before foot links)
    g1_root.append(right_attach)
    for elem in right_elements:
        g1_root.append(elem)

    g1_root.append(left_attach)
    for elem in left_elements:
        g1_root.append(elem)

    # Step 6: Update robot name
    g1_root.set("name", "g1_xhand")

    # Step 7: Write output
    _indent_xml(g1_root)
    g1_tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"Combined URDF written to: {output_path}")

    # Print summary
    joint_count = len(g1_root.findall("joint"))
    link_count = len(g1_root.findall("link"))
    revolute_count = sum(
        1 for j in g1_root.findall("joint") if j.get("type") == "revolute"
    )
    print(f"  Links: {link_count}, Joints: {joint_count}, Revolute: {revolute_count}")


def _create_attachment_joint(
    name: str, parent: str, child: str, origin: dict
) -> ET.Element:
    """
    Create a fixed joint element attaching XHand root to G1 wrist.

    Args:
        name: Joint name.
        parent: Parent link name (G1 wrist).
        child: Child link name (XHand root).
        origin: Dict with 'xyz' and 'rpy' strings.

    Returns:
        ET.Element: The joint XML element.
    """
    joint = ET.Element("joint", name=name, type="fixed")
    ET.SubElement(joint, "origin", xyz=origin["xyz"], rpy=origin["rpy"])
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)
    return joint


def _scale_hand_masses(elements: list, target_total_mass: float) -> None:
    """
    Scale link masses so the total hand mass matches measured weight.

    Args:
        elements: List of XML elements (links and joints).
        target_total_mass: Desired total mass in kg.
    """
    links_with_mass = []
    current_total = 0.0

    for elem in elements:
        if elem.tag == "link":
            inertial = elem.find("inertial")
            if inertial is not None:
                mass_elem = inertial.find("mass")
                if mass_elem is not None:
                    m = float(mass_elem.get("value", "0"))
                    links_with_mass.append((mass_elem, m))
                    current_total += m

    if current_total > 0 and abs(current_total - target_total_mass) > 0.001:
        scale = target_total_mass / current_total
        print(
            f"  Scaling hand mass: {current_total:.4f}kg -> "
            f"{target_total_mass:.4f}kg (scale={scale:.4f})"
        )
        for mass_elem, original_mass in links_with_mass:
            new_mass = original_mass * scale
            mass_elem.set("value", f"{new_mass:.8f}")


def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    """
    Add indentation to XML for readability.

    Args:
        elem: XML element to indent.
        level: Current indentation level.
    """
    indent = "\n" + "    " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "    "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent


def find_default_paths() -> dict:
    """
    Find default paths for G1 and XHand URDFs.

    Returns:
        dict: Keys 'g1', 'xhand_right', 'xhand_left' with Path values.

    Raises:
        FileNotFoundError: If required URDF files are not found.
    """
    paths = {}

    # G1 URDF from system install
    g1_candidates = [
        Path("/opt/ros/humble/share/unitree_description/urdf/g1/main.urdf"),
        Path("/opt/ros/jazzy/share/unitree_description/urdf/g1/main.urdf"),
    ]
    for p in g1_candidates:
        if p.exists():
            paths["g1"] = p
            break

    if "g1" not in paths:
        raise FileNotFoundError(
            "G1 URDF not found. Install unitree_description or specify --g1-urdf"
        )

    # XHand URDFs bundled inside g1_xhand_description/urdf/xhand_source/
    script_dir = Path(__file__).resolve().parent
    pkg_dir = script_dir.parent

    xhand_right = pkg_dir / "urdf" / "xhand_source" / "xhand_right.urdf"
    xhand_left = pkg_dir / "urdf" / "xhand_source" / "xhand_left.urdf"

    if not xhand_right.exists():
        raise FileNotFoundError(f"XHand right URDF not found at: {xhand_right}")
    if not xhand_left.exists():
        raise FileNotFoundError(f"XHand left URDF not found at: {xhand_left}")

    paths["xhand_right"] = xhand_right
    paths["xhand_left"] = xhand_left

    return paths


def main():
    """Entry point for URDF combination."""
    parser = argparse.ArgumentParser(
        description="Combine G1 URDF with XHand dexterous hand URDFs"
    )
    parser.add_argument("--g1-urdf", type=str, help="Path to G1 main.urdf")
    parser.add_argument("--xhand-right", type=str, help="Path to xhand_right.urdf")
    parser.add_argument("--xhand-left", type=str, help="Path to xhand_left.urdf")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for combined URDF",
    )
    parser.add_argument(
        "--mesh-package",
        type=str,
        default="g1_xhand_description",
        help="ROS package name for mesh references",
    )
    parser.add_argument(
        "--measured-mass",
        type=float,
        default=None,
        help="Measured hand mass in kg (per hand). Scales URDF masses to match.",
    )

    args = parser.parse_args()

    # Find default paths if not specified
    if not all([args.g1_urdf, args.xhand_right, args.xhand_left]):
        try:
            defaults = find_default_paths()
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

    g1_path = args.g1_urdf or str(defaults["g1"])
    right_path = args.xhand_right or str(defaults["xhand_right"])
    left_path = args.xhand_left or str(defaults["xhand_left"])

    # Default output path
    if args.output is None:
        script_dir = Path(__file__).resolve().parent
        output = script_dir.parent / "urdf" / "g1_xhand" / "main.urdf"
    else:
        output = Path(args.output)

    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"G1 URDF:     {g1_path}")
    print(f"XHand Right: {right_path}")
    print(f"XHand Left:  {left_path}")
    print(f"Output:      {output}")
    if args.measured_mass:
        print(f"Target mass: {args.measured_mass}kg per hand")
    print()

    combine_urdf(
        g1_urdf_path=g1_path,
        xhand_right_path=right_path,
        xhand_left_path=left_path,
        output_path=str(output),
        mesh_package=args.mesh_package,
        measured_hand_mass=args.measured_mass,
    )


if __name__ == "__main__":
    main()
