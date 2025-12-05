"""
Diagnose calibration issues with IMU data
"""
import torch
import numpy as np
import pandas as pd
from mobileposer.articulate.math import quaternion_to_rotation_matrix

# Load calibration frames (40-80)
wrist_df = pd.read_csv('../../IMU-recordings/Ido-iPhone/test4-lw-rp/WristMotion.csv')
acc_df = pd.read_csv('../../IMU-recordings/Ido-iPhone/test4-lw-rp/Accelerometer.csv')
ori_df = pd.read_csv('../../IMU-recordings/Ido-iPhone/test4-lw-rp/Orientation.csv')

# Skip first 3.5 seconds (352 frames for wrist, 348 for phone)
wrist_calib = wrist_df.iloc[352+40:352+80]
phone_acc_calib = acc_df.iloc[348+40:348+80]
phone_ori_calib = ori_df.iloc[348+40:348+80]

print("=== Calibration Frame Analysis (T-Pose) ===\n")

# Check wrist acceleration (should show gravity direction)
# Apple Watch separates gravity and user acceleration
wrist_gravity = wrist_calib[['gravityX', 'gravityY', 'gravityZ']].values
wrist_user_acc = wrist_calib[['accelerationX', 'accelerationY', 'accelerationZ']].values
wrist_acc = wrist_gravity + wrist_user_acc  # Total acceleration
wrist_acc_mean = wrist_acc.mean(axis=0)
wrist_acc_std = wrist_acc.std(axis=0)

print("Wrist Acceleration (in g):")
print(f"  Mean: X={wrist_acc_mean[0]:.3f}, Y={wrist_acc_mean[1]:.3f}, Z={wrist_acc_mean[2]:.3f}")
print(f"  Std:  X={wrist_acc_std[0]:.3f}, Y={wrist_acc_std[1]:.3f}, Z={wrist_acc_std[2]:.3f}")
print(f"  Magnitude: {np.linalg.norm(wrist_acc_mean):.3f}g")
print(f"  Expected: ~1g pointing down (gravity)")
print()

# Check phone acceleration
phone_acc = phone_acc_calib[['x', 'y', 'z']].values
phone_acc_mean = phone_acc.mean(axis=0)
phone_acc_std = phone_acc.std(axis=0)

print("Phone Acceleration (in g):")
print(f"  Mean: X={phone_acc_mean[0]:.3f}, Y={phone_acc_mean[1]:.3f}, Z={phone_acc_mean[2]:.3f}")
print(f"  Std:  X={phone_acc_std[0]:.3f}, Y={phone_acc_std[1]:.3f}, Z={phone_acc_std[2]:.3f}")
print(f"  Magnitude: {np.linalg.norm(phone_acc_mean):.3f}g")
print()

# Check if person was moving during calibration (std should be low)
print("Movement Check:")
print(f"  Wrist movement (std): {wrist_acc_std.mean():.4f}g")
print(f"  Phone movement (std): {phone_acc_std.mean():.4f}g")
if wrist_acc_std.mean() > 0.05 or phone_acc_std.mean() > 0.05:
    print("  ⚠️  WARNING: High movement detected! Should be still during T-pose calibration")
else:
    print("  ✓ Good - minimal movement")
print()

# Check orientation stability
wrist_quat = wrist_calib[['quaternionW', 'quaternionX', 'quaternionY', 'quaternionZ']].values
phone_quat = phone_ori_calib[['qw', 'qx', 'qy', 'qz']].values

wrist_quat_std = wrist_quat.std(axis=0)
phone_quat_std = phone_quat.std(axis=0)

print("Orientation Stability:")
print(f"  Wrist quat std: {wrist_quat_std.mean():.4f}")
print(f"  Phone quat std: {phone_quat_std.mean():.4f}")
if wrist_quat_std.mean() > 0.01 or phone_quat_std.mean() > 0.01:
    print("  ⚠️  WARNING: Orientation changing during calibration!")
else:
    print("  ✓ Good - stable orientation")
print()

# Convert to rotation matrices to see orientation
wrist_quat_mean = torch.from_numpy(wrist_quat.mean(axis=0)).float()
phone_quat_mean = torch.from_numpy(phone_quat.mean(axis=0)).float()

wrist_rot = quaternion_to_rotation_matrix(wrist_quat_mean.unsqueeze(0)).squeeze().numpy()
phone_rot = quaternion_to_rotation_matrix(phone_quat_mean.unsqueeze(0)).squeeze().numpy()

print("Wrist Rotation Matrix (T-pose):")
print(wrist_rot)
print()

print("Phone Rotation Matrix (T-pose):")
print(phone_rot)
print()

# Check if gravity aligns with expected direction
print("=== Gravity Direction Check ===")
wrist_gravity_dir = wrist_acc_mean / np.linalg.norm(wrist_acc_mean)
phone_gravity_dir = phone_acc_mean / np.linalg.norm(phone_acc_mean)

print(f"Wrist gravity direction: {wrist_gravity_dir}")
print(f"Phone gravity direction: {phone_gravity_dir}")
print()

# Expected: In T-pose, both should show gravity pointing down
# Wrist: arm extended horizontally, watch face up -> gravity in -Y or -Z
# Phone: in pocket, screen facing body -> gravity direction depends on pocket orientation

print("=== Recommendations ===")
if wrist_acc_std.mean() > 0.05 or phone_acc_std.mean() > 0.05:
    print("1. ⚠️  Re-record with person completely still during calibration frames")
if np.abs(np.linalg.norm(wrist_acc_mean) - 1.0) > 0.2:
    print("2. ⚠️  Wrist acceleration magnitude unusual - check sensor")
if np.abs(np.linalg.norm(phone_acc_mean) - 1.0) > 0.2:
    print("3. ⚠️  Phone acceleration magnitude unusual - check sensor")
