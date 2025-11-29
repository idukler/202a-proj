"""
Offline IMU processing - exact same logic as live demo but reading from CSV files
"""

import os
import torch
import numpy as np
import pandas as pd
from argparse import ArgumentParser

from articulate.math import *
from mobileposer.models import *
from mobileposer.utils.model_utils import *
from mobileposer.config import *
from mobileposer.viewer import Viewer


def load_csv_data(acc_file, ori_file, n_imus=5, gravity_file=None):
    """
    Load acceleration and orientation data from CSV files.
    
    Args:
        acc_file: Path to Acceleration.csv (user acceleration)
        ori_file: Path to Orientation.csv
        n_imus: Number of IMUs (default 5)
        gravity_file: Optional path to Gravity.csv. If None, gravity is computed from orientation.
    
    Returns:
        raw_ori: [N, n_imus, 4] quaternions (qw, qx, qy, qz)
        raw_acc: [N, n_imus, 3] accelerations (x, y, z) - total acceleration including gravity
        timestamps: [N] timestamps in nanoseconds
    """
    print(f"Loading acceleration data from {acc_file}...")
    acc_df = pd.read_csv(acc_file)
    
    print(f"Loading orientation data from {ori_file}...")
    ori_df = pd.read_csv(ori_file)
    
    # Check that both files have the same number of rows
    if len(acc_df) != len(ori_df):
        raise ValueError(f"Acceleration ({len(acc_df)} rows) and orientation ({len(ori_df)} rows) files must have same length")
    
    n_frames = len(acc_df)
    print(f"Loaded {n_frames} frames")
    
    # Extract timestamps (assuming they match between files)
    timestamps = torch.from_numpy(acc_df['time'].values).long()
    
    # For now, assuming single IMU data - we'll replicate it for n_imus
    # If you have multiple IMUs, you'll need to modify this section
    
    # Extract user acceleration: x, y, z columns (without gravity)
    user_acc = acc_df[['x', 'y', 'z']].values  # [N, 3]
    
    # Extract quaternion: qw, qx, qy, qz columns
    quat_single = ori_df[['qw', 'qx', 'qy', 'qz']].values  # [N, 4]
    
    # Get gravity component
    if gravity_file is not None:
        # Option 1: Load gravity from separate file (PREFERRED if available)
        print(f"Loading gravity data from {gravity_file}...")
        gravity_df = pd.read_csv(gravity_file)
        if len(gravity_df) != n_frames:
            raise ValueError(f"Gravity file has {len(gravity_df)} rows but expected {n_frames}")
        sensor_gravity = gravity_df[['x', 'y', 'z']].values  # Assuming same column names
    else:
        # Option 2: Compute gravity from orientation quaternion
        print("Computing gravity from orientation quaternion...")
        from scipy.spatial.transform import Rotation
        rotations = Rotation.from_quat(quat_single[:, [1, 2, 3, 0]])  # scipy uses [x,y,z,w]
        global_gravity = np.array([0, 0, -1.0])  # -1g in z direction (down)
        sensor_gravity = rotations.apply(global_gravity)  # Rotate to sensor frame
    
    # Total acceleration = user acceleration + gravity
    acc_single = user_acc + sensor_gravity
    print(f"  Total acceleration magnitude: {np.linalg.norm(acc_single, axis=1).mean():.3f}g (expected ~1g)")
    
    # Create arrays for n_imus (replicating single IMU for now)
    # TODO: If you have multiple IMU files, load each separately here
    raw_acc = np.zeros((n_frames, n_imus, 3))
    raw_ori = np.zeros((n_frames, n_imus, 4))
    
    # Initialize all quaternions to identity [qw=1, qx=0, qy=0, qz=0]
    raw_ori[:, :, 0] = 1.0  # Set qw to 1 for identity quaternion
    
    # Phone data goes to index 3 (right pocket)
    # Index mapping: 0=left wrist, 1=right wrist, 2=left pocket, 3=right pocket, 4=head
    raw_acc[:, 3, :] = acc_single
    raw_ori[:, 3, :] = quat_single
    
    # If you have separate files for each IMU, load them here:
    # raw_acc[:, 1, :] = pd.read_csv('imu2_acc.csv')[['x', 'y', 'z']].values
    # raw_ori[:, 1, :] = pd.read_csv('imu2_ori.csv')[['qw', 'qx', 'qy', 'qz']].values
    # etc.
    
    print(f"Warning: Currently loading single IMU data. You need {n_imus} IMUs for full body tracking.")
    print(f"Raw acceleration shape: {raw_acc.shape}")
    print(f"Raw orientation shape: {raw_ori.shape}")
    
    return torch.from_numpy(raw_ori).float(), torch.from_numpy(raw_acc).float(), timestamps


def load_combined_csv_data(wristmotion_file, acc_file, ori_file, n_imus=5, gravity_file=None, 
                           acc_file2=None, ori_file2=None, gravity_file2=None, skip_seconds=3.5):
    """
    Load WristMotion.csv (watch) and phone Acceleration/Orientation CSV files.
    Supports optional second phone for both pockets.
    
    Args:
        wristmotion_file: Path to WristMotion.csv (watch data, optional)
        acc_file: Path to Acceleration.csv (phone 1 - user acceleration)
        ori_file: Path to Orientation.csv (phone 1)
        n_imus: Number of IMUs (default 5)
        gravity_file: Optional path to phone 1 Gravity.csv
        acc_file2: Optional path to Acceleration.csv (phone 2 for second pocket)
        ori_file2: Optional path to Orientation.csv (phone 2)
        gravity_file2: Optional path to phone 2 Gravity.csv
        skip_seconds: Number of seconds to skip at the beginning (default 3.5)
    
    Returns:
        raw_ori: [N, n_imus, 4] quaternions (qw, qx, qy, qz)
        raw_acc: [N, n_imus, 3] accelerations (x, y, z) - total acceleration including gravity
        timestamps: [N] timestamps
    """
    # Load wrist data
    print(f"Loading WristMotion data from {wristmotion_file}...")
    wrist_df = pd.read_csv(wristmotion_file)
    print(f"Loaded {len(wrist_df)} frames from wrist")
    
    # Load phone data using existing function
    phone_ori, phone_acc, phone_timestamps = load_csv_data(acc_file, ori_file, n_imus, gravity_file)
    print(f"Loaded {len(phone_ori)} frames from phone")
    
    # Skip first 3.5 seconds based on timestamps
    wrist_start_time = wrist_df['time'].values[0]
    phone_start_time = phone_timestamps[0].item()
    
    skip_ns = int(skip_seconds * 1e9)  # Convert to nanoseconds
    
    # Find indices after skipping
    wrist_skip_idx = np.searchsorted(wrist_df['time'].values, wrist_start_time + skip_ns)
    phone_skip_idx = np.searchsorted(phone_timestamps.numpy(), phone_start_time + skip_ns)
    
    print(f"Skipping first {skip_seconds}s: wrist starts at frame {wrist_skip_idx}, phone starts at frame {phone_skip_idx}")
    
    # Trim to skip initial frames
    wrist_df = wrist_df.iloc[wrist_skip_idx:].reset_index(drop=True)
    phone_ori = phone_ori[phone_skip_idx:]
    phone_acc = phone_acc[phone_skip_idx:]
    phone_timestamps = phone_timestamps[phone_skip_idx:]
    
    # Trim to minimum length
    min_frames = min(len(wrist_df), len(phone_ori))
    print(f"Trimming to {min_frames} frames (wrist: {len(wrist_df)}, phone: {len(phone_ori)})")
    
    wrist_df = wrist_df.iloc[:min_frames]
    phone_ori = phone_ori[:min_frames]
    phone_acc = phone_acc[:min_frames]
    phone_timestamps = phone_timestamps[:min_frames]
    
    # Extract wrist data
    timestamps = torch.from_numpy(wrist_df['time'].values).long()
    # Use gravity + user acceleration for total acceleration (Apple Watch separates them)
    wrist_gravity = wrist_df[['gravityX', 'gravityY', 'gravityZ']].values
    wrist_user_acc = wrist_df[['accelerationX', 'accelerationY', 'accelerationZ']].values
    wrist_acc = wrist_gravity + wrist_user_acc  # Total acceleration = gravity + user acceleration
    wrist_quat = wrist_df[['quaternionW', 'quaternionX', 'quaternionY', 'quaternionZ']].values  # [N, 4]
    
    # Combine: wrist at index 0 (left wrist), phone 1 already at index 3 (right pocket)
    raw_acc = phone_acc.clone()
    raw_ori = phone_ori.clone()
    
    # Wrist data goes to index 0 (left wrist)
    if wristmotion_file:
        raw_acc[:, 0, :] = torch.from_numpy(wrist_acc).float()
        raw_ori[:, 0, :] = torch.from_numpy(wrist_quat).float()
    
    # Load second phone if provided (for left pocket - index 2)
    if acc_file2 and ori_file2:
        print(f"\nLoading second phone data...")
        phone2_ori, phone2_acc, phone2_timestamps = load_csv_data(acc_file2, ori_file2, n_imus, gravity_file2)
        print(f"Loaded {len(phone2_ori)} frames from phone 2")
        
        # Align and trim phone 2 data
        phone2_start_time = phone2_timestamps[0].item()
        phone2_skip_idx = np.searchsorted(phone2_timestamps.numpy(), phone2_start_time + skip_ns)
        phone2_ori = phone2_ori[phone2_skip_idx:]
        phone2_acc = phone2_acc[phone2_skip_idx:]
        
        # Trim to match other data
        min_frames = min(min_frames, len(phone2_ori))
        raw_acc = raw_acc[:min_frames]
        raw_ori = raw_ori[:min_frames]
        phone2_ori = phone2_ori[:min_frames]
        phone2_acc = phone2_acc[:min_frames]
        timestamps = timestamps[:min_frames]
        
        # Phone 2 goes to index 2 (left pocket)
        raw_acc[:, 2, :] = phone2_acc[:, 3, :]  # Phone 2 was loaded at index 3, move to index 2
        raw_ori[:, 2, :] = phone2_ori[:, 3, :]
        print(f"Added phone 2 data to index 2 (left pocket)")
    
    # Verify no zero quaternions remain
    quat_norms = raw_ori.norm(dim=-1)
    if (quat_norms < 0.01).any():
        print(f"WARNING: Found {(quat_norms < 0.01).sum().item()} near-zero quaternions after combining data")
    
    print(f"Combined data shape - acc: {raw_acc.shape}, ori: {raw_ori.shape}")
    return raw_ori, raw_acc, timestamps


def get_mean_measurement_from_buffer(raw_ori, raw_acc, start_idx, num_frames):
    """
    Get mean measurements over a window of frames (equivalent to get_mean_measurement_of_n_second).
    
    Args:
        raw_ori: Raw orientation quaternions [N, n_imus, 4]
        raw_acc: Raw accelerations [N, n_imus, 3]
        start_idx: Starting frame index
        num_frames: Number of frames to average
    
    Returns:
        Mean quaternion [n_imus, 4] and acceleration [n_imus, 3]
    """
    end_idx = min(start_idx + num_frames, raw_ori.shape[0])
    q = raw_ori[start_idx:end_idx].mean(dim=0)
    a = raw_acc[start_idx:end_idx].mean(dim=0)
    return q, a


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--wristmotion-csv', type=str, default=None,
                       help='Path to WristMotion.csv file (alternative to --acc-csv and --ori-csv)')
    parser.add_argument('--acc-csv', type=str, default=None, 
                       help='Path to Acceleration.csv file')
    parser.add_argument('--ori-csv', type=str, default=None,
                       help='Path to Orientation.csv file')
    parser.add_argument('--gravity-csv', type=str, default=None,
                       help='Path to Gravity.csv file (optional, computed from orientation if not provided)')
    parser.add_argument('--acc-csv2', type=str, default=None,
                       help='Path to second phone Acceleration.csv (for left pocket)')
    parser.add_argument('--ori-csv2', type=str, default=None,
                       help='Path to second phone Orientation.csv (for left pocket)')
    parser.add_argument('--gravity-csv2', type=str, default=None,
                       help='Path to second phone Gravity.csv (optional)')
    parser.add_argument('--model', type=str, default=paths.weights_file,
                       help='Path to model weights')
    parser.add_argument('--combo', type=str, default='lw_rp',
                       help='Device combination (lw_rp, lw, rp, etc.)')
    parser.add_argument('--n-imus', type=int, default=5,
                       help='Number of IMUs (default: 5 for left_arm, right_arm, left_leg, right_leg, head)')
    parser.add_argument('--with-tran', action='store_true',
                       help='Visualize with translation')
    parser.add_argument('--save-processed', type=str, default=None,
                       help='Optional: save processed poses to file')
    parser.add_argument('--calib-frames', type=int, default=80,
                       help='Number of frames to use for calibration (default: 80, ~40 per step)')
    args = parser.parse_args()
    
    # Check valid combo
    if args.combo not in amass.combos.keys():
        raise ValueError(f"Invalid combo: {args.combo}. Must be one of {list(amass.combos.keys())}")
    
    # Specify device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load CSV data
    if args.wristmotion_csv and args.acc_csv and args.ori_csv:
        # Load wrist + phone(s) data
        print(f"\nLoading combined data (wrist + phone)...")
        raw_ori_all, raw_acc_all, timestamps = load_combined_csv_data(
            args.wristmotion_csv, args.acc_csv, args.ori_csv, args.n_imus, args.gravity_csv,
            args.acc_csv2, args.ori_csv2, args.gravity_csv2)
    elif args.acc_csv and args.ori_csv:
        # Load phone data only (can be 1 or 2 phones)
        if args.acc_csv2 and args.ori_csv2:
            print(f"\nLoading two phones data...")
            raw_ori_all, raw_acc_all, timestamps = load_combined_csv_data(
                None, args.acc_csv, args.ori_csv, args.n_imus, args.gravity_csv,
                args.acc_csv2, args.ori_csv2, args.gravity_csv2)
        else:
            print(f"\nLoading phone data only...")
            raw_ori_all, raw_acc_all, timestamps = load_csv_data(args.acc_csv, args.ori_csv, args.n_imus, args.gravity_csv)
    else:
        raise ValueError("Must provide both --acc-csv and --ori-csv (and optionally --wristmotion-csv)")
    
    n_imus = args.n_imus
    total_frames = raw_ori_all.shape[0]
    
    # Convert acceleration from g to m/s^2 if needed
    # The live demo expects acceleration in m/s^2, CSVs might be in g
    # Check the magnitude - if it's around 1.0, it's in g, if it's around 9.8, it's in m/s^2
    acc_magnitude = raw_acc_all.norm(dim=-1).mean()
    if acc_magnitude < 2.0:  # Likely in g
        print(f"Acceleration appears to be in g (magnitude: {acc_magnitude:.2f}), converting to m/s^2...")
        raw_acc_all = raw_acc_all * 9.8
    else:
        print(f"Acceleration appears to be in m/s^2 (magnitude: {acc_magnitude:.2f})")
    
    # ============== CALIBRATION (same as live demo) ==============
    
    calib_frames_per_step = args.calib_frames // 2
    
    # Step 1: Align IMU to SMPL body frame
    print(f'\nStep 1: Aligning IMU 1 to body reference frame...')
    print(f'Using frames 0-{calib_frames_per_step} for IMU alignment...')
    oris = get_mean_measurement_from_buffer(raw_ori_all, raw_acc_all, 0, calib_frames_per_step)[0][0]
    smpl2imu = quaternion_to_rotation_matrix(oris).view(3, 3).t()
    print('\tFinished IMU alignment.')
    
    # Step 2: Bone and acceleration offset calibration (T-pose)
    print(f'\nStep 2: Bone and acceleration offset calibration (T-pose)...')
    print(f'Using frames {calib_frames_per_step}-{args.calib_frames} for T-pose calibration...')
    oris, accs = get_mean_measurement_from_buffer(raw_ori_all, raw_acc_all, calib_frames_per_step, calib_frames_per_step)
    oris = quaternion_to_rotation_matrix(oris)
    device2bone = smpl2imu.matmul(oris).transpose(1, 2).matmul(torch.eye(3))
    acc_offsets = smpl2imu.matmul(accs.unsqueeze(-1))   # [num_imus, 3, 1], already in global inertial frame
    print('\tFinished calibration.')
    
    # Skip calibration frames
    print(f'\nProcessing {total_frames - args.calib_frames} frames after calibration...')
    
    # Load model
    print('\nLoading model...')
    model = load_model(args.model)
    model = model.to(device)
    model.eval()
    
    # Move tensors to device
    smpl2imu = smpl2imu.to(device)
    device2bone = device2bone.to(device)
    acc_offsets = acc_offsets.to(device)
    
    # ============== PROCESS FRAMES (same as live demo loop) ==============
    
    poses = []
    trans = []
    
    print('\nProcessing frames...')
    for frame_idx in range(args.calib_frames, total_frames):
        # Get current frame (buffer_len=1 in live demo)
        ori_raw = raw_ori_all[frame_idx:frame_idx+1].to(device)  # [1, n_imus, 4]
        acc_raw = raw_acc_all[frame_idx:frame_idx+1].to(device)  # [1, n_imus, 3]
        
        # Calibration (exact same as live demo)
        ori_raw = quaternion_to_rotation_matrix(ori_raw).view(-1, n_imus, 3, 3)
        
        # Debug: Check for NaN after quaternion conversion
        if frame_idx == args.calib_frames and torch.isnan(ori_raw).any():
            print(f"\nWARNING: NaN in ori_raw after quaternion_to_rotation_matrix!")
            print(f"  Input quaternion: {raw_ori_all[frame_idx:frame_idx+1]}")
        
        glb_acc = (smpl2imu.matmul(acc_raw.view(-1, n_imus, 3, 1)) - acc_offsets).view(-1, n_imus, 3)
        glb_ori = smpl2imu.matmul(ori_raw).matmul(device2bone)
        
        # Normalization (exact same as live demo)
        _acc = glb_acc.view(-1, 5, 3)[:, [1, 4, 3, 0, 2]] / amass.acc_scale
        _ori = glb_ori.view(-1, 5, 3, 3)[:, [1, 4, 3, 0, 2]]
        acc = torch.zeros_like(_acc)
        ori = torch.zeros_like(_ori)
        
        # Device combo - use available IMUs and mirror for missing symmetric pairs
        c = amass.combos[args.combo]
        acc[:, c] = _acc[:, c] 
        ori[:, c] = _ori[:, c]
        
        # Mirror symmetric body parts if only one side is available
        # This allows the model to infer missing limbs from available ones
        if 0 in c and 1 not in c:  # Have left wrist, missing right wrist
            acc[:, 1] = _acc[:, 0]  # Mirror left to right
            ori[:, 1] = _ori[:, 0]
        elif 1 in c and 0 not in c:  # Have right wrist, missing left wrist
            acc[:, 0] = _acc[:, 1]  # Mirror right to left
            ori[:, 0] = _ori[:, 1]
        
        if 2 in c and 3 not in c:  # Have left pocket, missing right pocket
            acc[:, 3] = _acc[:, 2]  # Mirror left to right
            ori[:, 3] = _ori[:, 2]
        elif 3 in c and 2 not in c:  # Have right pocket, missing left pocket
            acc[:, 2] = _acc[:, 3]  # Mirror right to left
            ori[:, 2] = _ori[:, 3]
        
        imu_input = torch.cat([acc.flatten(1), ori.flatten(1)], dim=1)
        
        # Predict pose and translation (exact same as live demo)
        with torch.no_grad():
            output = model.forward_online(imu_input.squeeze(0), [imu_input.shape[0]])
            pred_pose = output[0]  # [24, 3, 3]
            pred_tran = output[2]  # [3]
        
        # Debug: Check for NaN
        if frame_idx == args.calib_frames and torch.isnan(pred_tran).any():
            print(f"\nWARNING: NaN detected in first frame translation!")
            print(f"  pred_tran: {pred_tran}")
            print(f"  imu_input has NaN: {torch.isnan(imu_input).any()}")
            print(f"  acc has NaN: {torch.isnan(acc).any()}")
            print(f"  ori has NaN: {torch.isnan(ori).any()}")
        
        # Convert rotation matrix to axis angle
        pose = rotation_matrix_to_axis_angle(pred_pose.view(1, 216)).view(72)
        tran = pred_tran
        
        # Keep track of data
        poses.append(pose)
        trans.append(tran)
        
        if (frame_idx - args.calib_frames) % 100 == 0:
            print(f'  Processed {frame_idx - args.calib_frames}/{total_frames - args.calib_frames} frames...', end='\r')
    
    print(f'\nProcessed all {total_frames - args.calib_frames} frames.')
    
    # Stack results
    poses = torch.stack(poses).cpu()
    trans = torch.stack(trans).cpu()
    
    # Save processed data if requested
    if args.save_processed:
        print(f'\nSaving processed data to {args.save_processed}...')
        output_data = {
            'pose': poses,
            'tran': trans,
            'combo': args.combo,
            'calibration': {
                'smpl2imu': smpl2imu.cpu(),
                'device2bone': device2bone.cpu(),
                'acc_offsets': acc_offsets.cpu()
            }
        }
        torch.save(output_data, args.save_processed)
    
    # Visualize with viewer
    print('\nPreparing visualization...')
    # Create viewer-compatible data structure
    viewer_data = {
        'pose': poses,
        'tran': trans if args.with_tran else torch.zeros_like(trans)
    }
    
    # Save temporarily for viewer
    temp_file = '/tmp/offline_pose_data.pt'
    torch.save(viewer_data, temp_file)
    
    print(f'\nPose data saved to: {temp_file}')
    print(f'  - pose shape: {poses.shape}')
    print(f'  - tran shape: {trans.shape}')
    
    # Try to visualize with viewer
    print('\nAttempting to start viewer...')
    try:
        # You may need to modify this based on your Viewer implementation
        v = Viewer(dataset='custom', seq_num=0, combo=args.combo)
        v.data = viewer_data  # Inject our data
        v.view(with_tran=args.with_tran)
    except Exception as e:
        print(f'Could not start viewer automatically: {e}')
        print(f'Please load the data from {temp_file} manually in your viewer.')
    
    print('\nDone!')