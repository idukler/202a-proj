"""
Run MobilePoser with Your Recorded IMU Data
============================================
This script runs MobilePoser inference on your recorded phone data.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import from mobileposer
sys.path.insert(0, str(Path(__file__).parent))

from mobileposer.config import paths, amass, model_config
from mobileposer.utils.model_utils import load_model


def run_inference_on_recorded_data(data_path, combo='rp', model_path=None):
    """
    Run MobilePoser inference on recorded IMU data.
    
    Args:
        data_path: Path to .npy or .csv file with 12D IMU data
                   Format: [accel_x, accel_y, accel_z, r11, r12, r13, r21, r22, r23, r31, r32, r33]
        combo: Device combination (default: 'rp' for right pocket)
        model_path: Path to model weights
    """
    print("=" * 70)
    print("MobilePoser - Running Inference on Your Recorded Data")
    print("=" * 70)
    
    # Check model weights
    if model_path is None:
        model_path = paths.weights_file
    
    if not Path(model_path).exists():
        print(f"\n❌ ERROR: Model weights not found at: {model_path}")
        print("\nPlease download the pretrained weights from:")
        print("https://uchicago.box.com/s/ey3y49srpo79propzvmjx0t8u3ael6cl")
        print(f"\nAnd place it at: {model_path}")
        return
    
    # Load data
    print(f"\n📂 Loading data from: {data_path}")
    data_path = Path(data_path)
    
    if data_path.suffix == '.npy':
        data = np.load(data_path)
    elif data_path.suffix == '.csv':
        import pandas as pd
        df = pd.read_csv(data_path)
        # Extract the 12D columns (skip time and seconds_elapsed)
        data = df[['accel_x', 'accel_y', 'accel_z', 
                   'r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33']].values
    else:
        print(f"❌ ERROR: Unsupported file format: {data_path.suffix}")
        return
    
    print(f"   Loaded {len(data)} frames")
    print(f"   Data shape: {data.shape}")
    
    # Validate combo
    if combo not in amass.combos:
        print(f"❌ ERROR: Invalid combo '{combo}'. Must be one of {list(amass.combos.keys())}")
        return
    
    combo_indices = amass.combos[combo]
    print(f"\n🤖 Initializing MobilePoser")
    print(f"   Device combo: {combo}")
    print(f"   Active IMU indices: {combo_indices}")
    print(f"   IMU mapping: 0=left-wrist, 1=right-wrist, 2=left-pocket, 3=right-pocket, 4=head")
    
    # Load model
    print(f"   Loading model from: {model_path}")
    model = load_model(model_path)
    model.eval()
    print("   ✓ Model loaded successfully")
    
    # Process data
    print(f"\n🏃 Processing {len(data)} frames...")
    
    # Extract acceleration and rotation matrices
    acc_data = data[:, :3]  # First 3 columns: accel_x, accel_y, accel_z
    rot_data = data[:, 3:].reshape(-1, 3, 3)  # Remaining 9 columns: rotation matrix
    
    # Calibration: use first 90 frames (3 seconds at 30Hz)
    cal_frames = min(90, len(data))
    print(f"   Calibrating with first {cal_frames} frames...")
    
    # Compute calibration parameters
    mean_acc = torch.tensor(acc_data[:cal_frames].mean(axis=0), dtype=torch.float32)
    mean_rot = torch.tensor(rot_data[:cal_frames].mean(axis=0), dtype=torch.float32)
    
    # SMPL to IMU transformation
    smpl2imu = mean_rot.t()
    device2bone = torch.eye(3).unsqueeze(0)
    acc_offsets = smpl2imu.matmul(mean_acc.unsqueeze(-1))
    
    print("   ✓ Calibration complete")
    
    # Run inference on all frames
    poses_list = []
    joints_list = []
    translations_list = []
    
    print(f"   Running inference...")
    
    with torch.no_grad():
        for i in range(len(data)):
            # Get current frame data
            acc = torch.tensor(acc_data[i], dtype=torch.float32)
            rot = torch.tensor(rot_data[i], dtype=torch.float32)
            
            # Transform to global frame
            glb_acc = (smpl2imu.matmul(acc.unsqueeze(-1)) - acc_offsets).squeeze(-1)
            glb_ori = smpl2imu.matmul(rot).matmul(device2bone.squeeze(0))
            
            # Normalize acceleration
            glb_acc = glb_acc / amass.acc_scale
            
            # Create input tensor for all 5 possible IMU locations
            acc_input = torch.zeros(5, 3)
            ori_input = torch.zeros(5, 3, 3)
            
            # Fill in the active IMU location(s)
            for idx in combo_indices:
                acc_input[idx] = glb_acc
                ori_input[idx] = glb_ori
            
            # Flatten and concatenate: [acc (15), ori (45)] = 60 dimensions
            imu_input = torch.cat([acc_input.flatten(), ori_input.flatten()])
            
            # Move to device
            imu_input = imu_input.to(model_config.device)
            
            # Inference
            pose, joints, translation, contact = model.forward_online(imu_input, [1])
            
            poses_list.append(pose.cpu())
            joints_list.append(joints.cpu())
            translations_list.append(translation.cpu())
            
            # Progress indicator
            if (i + 1) % 50 == 0 or (i + 1) == len(data):
                print(f"      Processed {i + 1}/{len(data)} frames", end='\r')
    
    print(f"\n   ✓ Inference complete!")
    
    # Stack results
    poses = torch.stack(poses_list)
    joints = torch.stack(joints_list)
    translations = torch.stack(translations_list)
    
    # Save results
    output_file = data_path.stem + "_mobileposer_results.pt"
    output_path = Path.cwd() / output_file
    
    print(f"\n💾 Saving results to: {output_path}")
    torch.save({
        'poses': poses,              # [N, 24, 3, 3] - SMPL pose parameters
        'joints': joints,            # [N, 24, 3] - 3D joint positions
        'translations': translations, # [N, 3] - Root translations
        'combo': combo,
        'num_frames': len(data),
        'data_file': str(data_path)
    }, output_path)
    print("   ✓ Results saved")
    
    # Print summary statistics
    print("\n📊 Summary Statistics:")
    print(f"   Total frames processed: {len(data)}")
    print(f"   Poses shape: {poses.shape}")
    print(f"   Joints shape: {joints.shape}")
    print(f"   Translations shape: {translations.shape}")
    
    translations_np = translations.numpy()
    print(f"\n   Translation range:")
    print(f"     X: [{translations_np[:, 0].min():.3f}, {translations_np[:, 0].max():.3f}] m")
    print(f"     Y: [{translations_np[:, 1].min():.3f}, {translations_np[:, 1].max():.3f}] m")
    print(f"     Z: [{translations_np[:, 2].min():.3f}, {translations_np[:, 2].max():.3f}] m")
    
    # Calculate total distance traveled
    diffs = np.diff(translations_np, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    total_distance = distances.sum()
    print(f"\n   Total distance traveled: {total_distance:.3f} m")
    
    print("\n" + "=" * 70)
    print("✅ Inference completed successfully!")
    print("=" * 70)
    
    return poses, joints, translations


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MobilePoser on your recorded IMU data")
    parser.add_argument('--data', type=str, 
                        default='../../sensor_data_12d.npy',
                        help='Path to .npy or .csv file with IMU data')
    parser.add_argument('--combo', type=str, default='rp',
                        help='Device combination (default: rp for right pocket)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model weights (default: use config path)')
    
    args = parser.parse_args()
    
    run_inference_on_recorded_data(args.data, combo=args.combo, model_path=args.model)
