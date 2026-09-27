#!/usr/bin/env python3
"""Convert yaw/pitch/roll (degrees) to a 3x3 rotation matrix and print
in YAML-like `R_camera2gimbal: [...]` format.

Usage examples:
  python3 scripts/euler_to_matrix.py --input "yaw-3.98 pitch-1.79 roll2.74 degree"
  python3 scripts/euler_to_matrix.py --yaw -3.98 --pitch -1.79 --roll 2.74
"""
import re
import argparse
import numpy as np


def parse_input(s: str):
    # extract first three signed numbers from string
    nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
    if len(nums) < 3:
        raise ValueError('expected at least three numbers in input')
    y, p, r = map(float, nums[:3])
    return y, p, r


def euler_to_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    # Use Z (yaw) -> Y (pitch) -> X (roll) order: R = Rz * Ry * Rx
    y = np.deg2rad(yaw_deg)
    p = np.deg2rad(pitch_deg)
    r = np.deg2rad(roll_deg)
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    R = Rz.dot(Ry).dot(Rx)
    return R


def format_yaml_matrix(R: np.ndarray) -> str:
    vals = R.flatten()
    parts = []
    for i, v in enumerate(vals):
        s = '{:.17g}'.format(float(v))
        if (i + 1) % 3 == 0 and i != len(vals) - 1:
            s += ',\n '
        else:
            s += ', '
        parts.append(s)
    body = ''.join(parts).rstrip(', ')
    return f'R_camera2gimbal: [{body}]'


def format_ascento_matrix(R: np.ndarray) -> str:
    # Reorder and flip signs to match the ascento.yaml convention observed in the repo
    # Desired order (from config): [r01, r02, r00, -r11, -r12, -r10, -r21, -r22, -r20]
    r = R
    vals = [
        r[0, 1], r[0, 2], r[0, 0],
        -r[1, 1], -r[1, 2], -r[1, 0],
        -r[2, 1], -r[2, 2], -r[2, 0],
    ]
    parts = []
    for i, v in enumerate(vals):
        s = '{:.17g}'.format(float(v))
        if (i + 1) % 3 == 0 and i != len(vals) - 1:
            s += ',\n '
        else:
            s += ', '
        parts.append(s)
    body = ''.join(parts).rstrip(', ')
    return f'R_camera2gimbal: [{body}]'


def main():
    parser = argparse.ArgumentParser(description='Euler to rotation matrix (Z-Y-X order)')
    parser.add_argument('--input', '-i', help='single string like "yaw-3.98 pitch-1.79 roll2.74 degree"')
    parser.add_argument('--yaw', type=float, help='yaw in degrees')
    parser.add_argument('--pitch', type=float, help='pitch in degrees')
    parser.add_argument('--roll', type=float, help='roll in degrees')
    parser.add_argument('--ascento', action='store_true', help='format output to match ascento.yaml ordering')
    args = parser.parse_args()

    if args.input:
        yaw, pitch, roll = parse_input(args.input)
    elif args.yaw is not None and args.pitch is not None and args.roll is not None:
        yaw, pitch, roll = args.yaw, args.pitch, args.roll
    else:
        parser.error('provide --input or all of --yaw --pitch --roll')

    R = euler_to_matrix(yaw, pitch, roll)
    if args.ascento:
        print(format_ascento_matrix(R))
    else:
        print(format_yaml_matrix(R))


if __name__ == '__main__':
    main()
