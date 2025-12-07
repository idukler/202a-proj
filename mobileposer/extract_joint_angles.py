"""
Extract SMPL pose data from .pt file and calculate joint angles.

This script:
1. Loads the saved .pt file from live_demo_phone_http.py
2. Extracts pose data (axis-angle format)
3. Computes joint positions using forward kinematics
4. Calculates joint angles (knee, hip, ankle) from joint positions
5. Exports data in JSON format for React app consumption
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path to import mobileposer modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from mobileposer.config import paths
import mobileposer.articulate as art


# SMPL joint indices (from pygame_visualizer.py)
JOINT_NAMES = [
    'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 
    'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck', 
    'L_Collar', 'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 
    'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
]

# Joint indices for angle calculations
PELVIS = 0
L_HIP = 1
R_HIP = 2
SPINE1 = 3
L_KNEE = 4
R_KNEE = 5
SPINE2 = 6
L_ANKLE = 7
R_ANKLE = 8
SPINE3 = 9
L_FOOT = 10
R_FOOT = 11
NECK = 12
HEAD = 15
L_SHOULDER = 16
R_SHOULDER = 17
L_ELBOW = 18
R_ELBOW = 19
L_WRIST = 20
R_WRIST = 21


def calculate_angle_between_vectors(v1, v2):
    """
    Calculate the angle (in degrees) between two 3D vectors.
    
    Args:
        v1: [3] vector
        v2: [3] vector
    
    Returns:
        angle in degrees
    """
    # Normalize vectors
    v1_norm = v1 / (torch.norm(v1) + 1e-8)
    v2_norm = v2 / (torch.norm(v2) + 1e-8)
    
    # Calculate dot product and clamp to [-1, 1] for numerical stability
    dot_product = torch.clamp(torch.dot(v1_norm, v2_norm), -1.0, 1.0)
    
    # Calculate angle in radians, then convert to degrees
    angle_rad = torch.acos(dot_product)
    angle_deg = torch.rad2deg(angle_rad)
    
    return angle_deg.item()


def determine_front_back_leg(joint_positions):
    """
    Determine which leg is in front based on forward position (Z-axis in SMPL).
    In SMPL coordinate system: X=left, Y=up, Z=forward.
    
    Args:
        joint_positions: [24, 3] tensor of joint positions
    
    Returns:
        tuple: (front_side, back_side) where both are 'left' or 'right'
    """
    # Compare foot positions in forward direction (Z-axis, index 2)
    l_foot_z = joint_positions[L_FOOT][2].item()
    r_foot_z = joint_positions[R_FOOT][2].item()
    
    # The foot more forward (larger Z) is the front leg
    if l_foot_z > r_foot_z:
        return 'left', 'right'
    else:
        return 'right', 'left'


def calculate_joint_angles(joint_positions):
    """
    Calculate joint angles from joint positions.
    
    Args:
        joint_positions: [24, 3] tensor of joint positions
    
    Returns:
        dict with front knee, back knee, back-to-head angle, and elbow angles
    """
    # Convert to numpy if needed
    if isinstance(joint_positions, torch.Tensor):
        joint_positions = joint_positions.cpu()
    
    # Calculate knee angles for both legs
    # Knee angle is the interior angle at the knee joint (hip->knee->ankle)
    # We need vectors pointing FROM the knee joint TO the adjacent joints
    # Then measure the angle between them to get the interior joint angle
    l_knee_to_hip = joint_positions[L_HIP] - joint_positions[L_KNEE]  # Vector from knee to hip
    l_knee_to_ankle = joint_positions[L_ANKLE] - joint_positions[L_KNEE]  # Vector from knee to ankle
    l_knee_angle = calculate_angle_between_vectors(l_knee_to_hip, l_knee_to_ankle)
    
    # Right knee angle
    r_knee_to_hip = joint_positions[R_HIP] - joint_positions[R_KNEE]  # Vector from knee to hip
    r_knee_to_ankle = joint_positions[R_ANKLE] - joint_positions[R_KNEE]  # Vector from knee to ankle
    r_knee_angle = calculate_angle_between_vectors(r_knee_to_hip, r_knee_to_ankle)
    
    # Determine which leg is front/back
    front_side, back_side = determine_front_back_leg(joint_positions)
    
    # Get front and back knee angles
    if front_side == 'left':
        front_knee_angle = l_knee_angle
        back_knee_angle = r_knee_angle
    else:
        front_knee_angle = r_knee_angle
        back_knee_angle = l_knee_angle
    
    # Back to head angle: angle of spine relative to vertical
    # Calculate angle between vertical (upward Y) and spine vector (pelvis->head)
    pelvis_to_head = joint_positions[HEAD] - joint_positions[PELVIS]
    vertical = torch.tensor([0.0, 1.0, 0.0])  # Upward vector
    back_to_head_angle = calculate_angle_between_vectors(vertical, pelvis_to_head)
    
    # Alternative: angle between spine segments (pelvis->spine3 and spine3->head)
    # This gives the spine curvature angle
    pelvis_to_spine3 = joint_positions[SPINE3] - joint_positions[PELVIS]
    spine3_to_head = joint_positions[HEAD] - joint_positions[SPINE3]
    spine_curvature_angle = calculate_angle_between_vectors(pelvis_to_spine3, spine3_to_head)
    
    # Left elbow angle: interior angle at elbow joint
    # Vectors pointing FROM elbow TO adjacent joints
    l_elbow_to_shoulder = joint_positions[L_SHOULDER] - joint_positions[L_ELBOW]
    l_elbow_to_wrist = joint_positions[L_WRIST] - joint_positions[L_ELBOW]
    l_elbow_angle = calculate_angle_between_vectors(l_elbow_to_shoulder, l_elbow_to_wrist)
    
    # Right elbow angle
    r_elbow_to_shoulder = joint_positions[R_SHOULDER] - joint_positions[R_ELBOW]
    r_elbow_to_wrist = joint_positions[R_WRIST] - joint_positions[R_ELBOW]
    r_elbow_angle = calculate_angle_between_vectors(r_elbow_to_shoulder, r_elbow_to_wrist)
    
    return {
        'frontKnee': {
            'angle': float(front_knee_angle),
            'side': front_side
        },
        'backKnee': {
            'angle': float(back_knee_angle),
            'side': back_side
        },
        'backToHead': {
            'angle': float(back_to_head_angle),
            'spineCurvature': float(spine_curvature_angle)
        },
        'elbow': {
            'left': float(l_elbow_angle),
            'right': float(r_elbow_angle)
        },
        # Also keep individual knee angles for reference
        'knee': {
            'left': float(l_knee_angle),
            'right': float(r_knee_angle)
        }
    }


def calculate_symmetry(left, right):
    """Calculate symmetry percentage between left and right angles."""
    if left == 0 and right == 0:
        return 100.0
    avg = (left + right) / 2
    # Handle edge case where avg is zero (e.g., left=5, right=-5)
    # If avg == 0, then left + right == 0, so right == -left
    # If left == right and avg == 0, then left == 0 (already handled above)
    # So if avg == 0, left != right, meaning values sum to zero but differ
    if abs(avg) < 1e-10:  # Use small epsilon to handle floating point precision
        # Values sum to zero but are different, return 0% symmetry
        return 0.0
    diff = abs(left - right)
    symmetry = max(0.0, 100.0 - (diff / avg) * 100.0)
    return round(symmetry, 1)


def extract_poses_from_pt_file(pt_file_path):
    """
    Extract pose data from .pt file and calculate joint angles.
    
    Args:
        pt_file_path: Path to the .pt file
    
    Returns:
        dict with time series data and joint angles
    """
    print(f"Loading data from {pt_file_path}...")
    
    # Load the .pt file
    data = torch.load(pt_file_path, map_location='cpu')
    
    # Check what keys are available
    print(f"Available keys in .pt file: {list(data.keys())}")
    
    # Extract pose data - it should be in 'actual_poses' key based on live_demo_phone_http.py
    if 'actual_poses' in data:
        poses = data['actual_poses']  # [num_frames, 72]
        trans = data.get('tran', None)
        
        # Debug translation data
        if trans is not None:
            print(f"  Found 'tran' key with shape: {trans.shape}, dtype: {trans.dtype}")
        else:
            print(f"  No 'tran' key found, will use zero translations")
        
        if trans is not None:
            # Handle different translation formats
            if len(trans.shape) == 0:
                # Scalar (shouldn't happen, but handle it)
                print(f"  Warning: Translation is scalar, using zeros")
                trans = torch.zeros(len(poses), 3)
            elif len(trans.shape) == 1:
                if trans.shape[0] == 3:
                    # Single translation vector [3], repeat for all frames
                    trans = trans.unsqueeze(0).repeat(len(poses), 1)
                elif trans.shape[0] == len(poses) * 3:
                    # Flattened format [num_frames * 3], reshape
                    trans = trans.view(len(poses), 3)
                else:
                    print(f"  Warning: Translation has unexpected 1D shape {trans.shape}, using zeros")
                    trans = torch.zeros(len(poses), 3)
            elif len(trans.shape) == 2:
                if trans.shape[1] == 3:
                    # [num_frames, 3] or [some_frames, 3]
                    if trans.shape[0] >= len(poses):
                        trans = trans[:len(poses)]  # Take first len(poses) frames
                    elif trans.shape[0] == 1:
                        # Single frame, repeat
                        trans = trans.repeat(len(poses), 1)
                    else:
                        print(f"  Warning: Translation shape {trans.shape} doesn't match {len(poses)} frames, using zeros")
                        trans = torch.zeros(len(poses), 3)
                else:
                    print(f"  Warning: Translation has unexpected 2D shape {trans.shape}, using zeros")
                    trans = torch.zeros(len(poses), 3)
            else:
                print(f"  Warning: Translation has unexpected shape {trans.shape}, using zeros")
                trans = torch.zeros(len(poses), 3)
        else:
            # Default translation (zero)
            trans = torch.zeros(len(poses), 3)
    elif 'pose' in data:
        poses = data['pose']  # Might be rotation matrices
        if poses.shape[-1] == 216:  # 24 joints * 9 (rotation matrix flattened)
            # Convert rotation matrices to axis-angle
            poses = poses.view(-1, 24, 3, 3)
            poses = art.math.rotation_matrix_to_axis_angle(poses).view(-1, 72)
        trans = data.get('tran', torch.zeros(len(poses), 3))
    else:
        raise ValueError(f"Could not find 'actual_poses' or 'pose' in .pt file. Available keys: {list(data.keys())}")
    
    # Validate that we have at least one frame
    if len(poses) == 0:
        raise ValueError(
            "The .pt file contains zero frames of pose data. "
            "Cannot extract joint angles from an empty dataset."
        )
    
    print(f"Found {len(poses)} frames of pose data")
    
    # Initialize SMPL model for forward kinematics
    smpl_file = paths.smpl_file
    if not os.path.exists(smpl_file):
        # Try relative path
        smpl_file = Path(__file__).parent / "smpl" / "basicmodel_m.pkl"
        if not os.path.exists(smpl_file):
            raise FileNotFoundError(f"SMPL model file not found at {paths.smpl_file} or {smpl_file}")
    
    print(f"Loading SMPL model from {smpl_file}...")
    device = torch.device('cpu')
    bodymodel = art.model.ParametricModel(str(smpl_file), device=device)
    
    # Process each frame
    time_series_data = []
    all_joint_angles = []
    
    print("Processing frames...")
    print(f"  Pose shape: {poses.shape}")
    print(f"  Translation shape: {trans.shape}")
    
    # Validate translation shape
    if len(trans.shape) == 0 or (len(trans.shape) == 1 and trans.shape[0] == 0):
        print("  Warning: Translation tensor is empty, using zero translations")
        trans = torch.zeros(len(poses), 3)
    elif len(trans.shape) == 1 and trans.shape[0] != 3:
        print(f"  Warning: Translation has unexpected shape {trans.shape}, reshaping...")
        if trans.shape[0] == len(poses) * 3:
            # Flattened format, reshape
            trans = trans.view(len(poses), 3)
        else:
            print(f"  Using zero translations as fallback")
            trans = torch.zeros(len(poses), 3)
    
    for i, pose in enumerate(poses):
        if i % 10 == 0:
            print(f"  Processing frame {i}/{len(poses)}")
        
        # Get translation for this frame
        if len(trans.shape) > 1:
            tran = trans[i]  # Should be [3]
        else:
            tran = trans[0] if len(trans) > 0 and trans.shape[0] >= 3 else torch.zeros(3)
        
        # Ensure translation is [3]
        if len(tran.shape) == 0 or tran.shape[0] != 3:
            print(f"  Warning: Invalid translation shape {tran.shape} for frame {i}, using zeros")
            tran = torch.zeros(3)
        
        # Ensure pose is the right shape [72]
        if pose.shape[0] != 72:
            print(f"  Warning: Unexpected pose shape {pose.shape} for frame {i}, expected [72]")
            continue
        
        # Convert axis-angle to rotation matrix for forward kinematics
        # Pose is [72] = [24, 3] axis-angle
        # Ensure pose is on CPU and float32
        pose = pose.cpu().float() if isinstance(pose, torch.Tensor) else torch.tensor(pose, dtype=torch.float32)
        pose_aa = pose.view(24, 3)
        
        # Convert to rotation matrices: [24, 3, 3]
        pose_rot = art.math.axis_angle_to_rotation_matrix(pose_aa)
        
        # Check for invalid values
        if torch.isnan(pose_rot).any() or torch.isinf(pose_rot).any():
            print(f"  Warning: Invalid values (NaN/Inf) in pose rotation matrices for frame {i}, skipping...")
            continue
        
        # Add batch dimension: [1, 24, 3, 3] (the format forward_kinematics expects)
        # Ensure tensor is contiguous for proper reshaping
        pose_rot_batch = pose_rot.unsqueeze(0).contiguous()
        
        # Ensure translation is correct shape: [1, 3] and on CPU
        if isinstance(tran, torch.Tensor):
            tran = tran.cpu().float()
            if len(tran.shape) == 1:
                tran_batch = tran.unsqueeze(0)  # [1, 3]
            else:
                tran_batch = tran
        else:
            tran_batch = torch.tensor(tran, dtype=torch.float32).unsqueeze(0) if len(tran) == 3 else torch.zeros(1, 3, dtype=torch.float32)
        
        # Forward kinematics to get joint positions
        # Expects [batch, 24, 3, 3] format
        try:
            pose_global, joint_positions = bodymodel.forward_kinematics(pose_rot_batch, shape=None, tran=tran_batch)
            
            # Debug first frame
            if i == 0:
                print(f"    Debug frame 0:")
                print(f"      pose_rot_batch shape: {pose_rot_batch.shape}")
                print(f"      tran_batch shape: {tran_batch.shape}")
                print(f"      joint_positions shape: {joint_positions.shape if joint_positions is not None else None}")
                if joint_positions is not None and joint_positions.shape[0] > 0:
                    print(f"      joint_positions[0] sample (first 3 joints): {joint_positions[0, :3]}")
                else:
                    print(f"      WARNING: joint_positions is empty or None!")
                    print(f"      pose_rot_batch min/max: {pose_rot_batch.min().item():.4f}/{pose_rot_batch.max().item():.4f}")
                    print(f"      pose_rot_batch contains NaN: {torch.isnan(pose_rot_batch).any().item()}")
                    print(f"      pose_rot_batch contains Inf: {torch.isinf(pose_rot_batch).any().item()}")
            
            # Check if joint_positions is valid
            if joint_positions is None:
                print(f"  Warning: None joint positions for frame {i}, skipping...")
                continue
            
            if joint_positions.shape[0] == 0:
                print(f"  Warning: Empty joint positions (shape {joint_positions.shape}) for frame {i}, skipping...")
                continue
            
            joint_positions = joint_positions[0]  # Get first (and only) frame: [24, 3]
            
            # Verify joint positions shape
            if joint_positions.shape[0] != 24:
                print(f"  Warning: Unexpected joint positions shape {joint_positions.shape} for frame {i}, expected [24, 3], skipping...")
                continue
                
        except Exception as e:
            print(f"  Error processing frame {i}: {e}")
            print(f"    pose_rot_batch shape: {pose_rot_batch.shape}")
            print(f"    tran_batch shape: {tran_batch.shape}")
            if i == 0:  # Only print full traceback for first error
                import traceback
                traceback.print_exc()
            continue
        
        # Calculate joint angles
        joint_angles = calculate_joint_angles(joint_positions)
        
        # Calculate symmetry for elbows
        joint_angles['elbow']['symmetry'] = calculate_symmetry(
            joint_angles['elbow']['left'],
            joint_angles['elbow']['right']
        )
        
        # Calculate symmetry for knees (left vs right, not front vs back)
        joint_angles['knee']['symmetry'] = calculate_symmetry(
            joint_angles['knee']['left'],
            joint_angles['knee']['right']
        )
        
        time_series_data.append({
            'timestamp': i,
            'jointAngles': joint_angles
        })
        
        all_joint_angles.append(joint_angles)
    
    # Check if we have any valid frames
    if len(all_joint_angles) == 0:
        raise ValueError(
            "No valid frames were processed! All frames returned empty joint positions. "
            "This might indicate an issue with the pose data format or the SMPL model."
        )
    
    print(f"\n  Successfully processed {len(all_joint_angles)}/{len(poses)} frames")
    
    # Calculate average angles
    avg_front_knee = np.mean([a['frontKnee']['angle'] for a in all_joint_angles])
    avg_back_knee = np.mean([a['backKnee']['angle'] for a in all_joint_angles])
    avg_back_to_head = np.mean([a['backToHead']['angle'] for a in all_joint_angles])
    avg_spine_curvature = np.mean([a['backToHead']['spineCurvature'] for a in all_joint_angles])
    avg_elbow_left = np.mean([a['elbow']['left'] for a in all_joint_angles])
    avg_elbow_right = np.mean([a['elbow']['right'] for a in all_joint_angles])
    avg_knee_left = np.mean([a['knee']['left'] for a in all_joint_angles])
    avg_knee_right = np.mean([a['knee']['right'] for a in all_joint_angles])
    
    # Determine most common front/back side
    front_sides = [a['frontKnee']['side'] for a in all_joint_angles]
    back_sides = [a['backKnee']['side'] for a in all_joint_angles]
    
    # Validate that we have sides to process (should be caught by check above, but double-check)
    if len(front_sides) == 0 or len(back_sides) == 0:
        raise ValueError(
            f"Cannot determine front/back sides: front_sides has {len(front_sides)} items, "
            f"back_sides has {len(back_sides)} items. This should not happen if all_joint_angles "
            f"is non-empty (has {len(all_joint_angles)} items)."
        )
    
    most_common_front = max(set(front_sides), key=front_sides.count)
    most_common_back = max(set(back_sides), key=back_sides.count)
    
    result = {
        'timeSeriesData': time_series_data,
        'jointAngles': {
            'frontKnee': {
                'angle': float(avg_front_knee),
                'side': most_common_front
            },
            'backKnee': {
                'angle': float(avg_back_knee),
                'side': most_common_back
            },
            'backToHead': {
                'angle': float(avg_back_to_head),
                'spineCurvature': float(avg_spine_curvature)
            },
            'elbow': {
                'left': float(avg_elbow_left),
                'right': float(avg_elbow_right),
                'symmetry': calculate_symmetry(avg_elbow_left, avg_elbow_right)
            },
            # Keep knee angles for backward compatibility
            'knee': {
                'left': float(avg_knee_left),
                'right': float(avg_knee_right),
                'symmetry': calculate_symmetry(avg_knee_left, avg_knee_right)
            }
        }
    }
    
    return result


def main():
    """Main function to extract and export joint angles."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract joint angles from .pt file')
    parser.add_argument('--input', '-i', type=str, 
                       default='phone_dev_1765133853.pt',
                       help='Path to input .pt file')
    parser.add_argument('--output', '-o', type=str,
                       default='joint_angles.json',
                       help='Path to output JSON file')
    parser.add_argument('--output-dir', type=str,
                       default=None,
                       help='Directory to save output (default: same as input file)')
    
    args = parser.parse_args()
    
    # Resolve input path
    input_path = Path(args.input)
    if not input_path.is_absolute():
        # Try relative to script directory first
        script_dir = Path(__file__).parent
        input_path = script_dir / input_path
        if not input_path.exists():
            # Try current working directory
            input_path = Path(args.input)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Determine output path
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / args.output
    else:
        output_path = input_path.parent / args.output
    
    # Extract joint angles
    result = extract_poses_from_pt_file(str(input_path))
    
    # Save to JSON
    print(f"\nSaving results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✓ Successfully extracted joint angles!")
    print(f"  - Total frames: {len(result['timeSeriesData'])}")
    print(f"  - Front knee angle: {result['jointAngles']['frontKnee']['angle']:.1f}° ({result['jointAngles']['frontKnee']['side']} leg)")
    print(f"  - Back knee angle: {result['jointAngles']['backKnee']['angle']:.1f}° ({result['jointAngles']['backKnee']['side']} leg)")
    print(f"  - Back to head angle: {result['jointAngles']['backToHead']['angle']:.1f}°")
    print(f"  - Spine curvature: {result['jointAngles']['backToHead']['spineCurvature']:.1f}°")
    print(f"  - Elbow angles: L={result['jointAngles']['elbow']['left']:.1f}°, R={result['jointAngles']['elbow']['right']:.1f}°")
    print(f"  - Elbow symmetry: {result['jointAngles']['elbow']['symmetry']:.1f}%")
    print(f"  - Output saved to: {output_path}")


if __name__ == '__main__':
    main()

