"""
Simple Test Script for MobilePoser with Custom Data
====================================================
This is a minimal script to test MobilePoser with your custom IMU data.

Usage:
    python test_custom_data.py --data your_data.csv

Expected CSV format:
    acc_x, acc_y, acc_z, ori_x, ori_y, ori_z, ori_w
    
    Where:
    - acc_x, acc_y, acc_z: Acceleration in m/s^2
    - ori_x, ori_y, ori_z, ori_w: Orientation quaternion (x, y, z, w)
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from argparse import ArgumentParser

from mobileposer.config import paths
from custom_imu_inference import CustomIMUProcessor


def test_with_csv(csv_file, combo='rp', model_path=None, visualize=False):
    """
    Test MobilePoser with a CSV file containing IMU data.
    
    Args:
        csv_file: Path to CSV file with IMU data
        combo: Device combination (default: 'rp' for right pocket)
        model_path: Path to model weights (default: use config path)
        visualize: Whether to visualize results (requires matplotlib)
    """
    print("=" * 60)
    print("MobilePoser Custom Data Test")
    print("=" * 60)
    
    # Check if weights file exists
    if model_path is None:
        model_path = paths.weights_file
    
    if not Path(model_path).exists():
        print(f"\n❌ ERROR: Model weights not found at: {model_path}")
        print("\nPlease download the pretrained weights from:")
        print("https://uchicago.box.com/s/ey3y49srpo79propzvmjx0t8u3ael6cl")
        print(f"\nAnd place it at: {model_path}")
        return
    
    # Load CSV data
    print(f"\n📂 Loading data from: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
        print(f"   Loaded {len(df)} frames")
    except Exception as e:
        print(f"❌ ERROR loading CSV: {e}")
        print("\nExpected CSV format:")
        print("acc_x, acc_y, acc_z, ori_x, ori_y, ori_z, ori_w")
        return
    
    # Validate columns
    required_cols = ['acc_x', 'acc_y', 'acc_z', 'ori_x', 'ori_y', 'ori_z', 'ori_w']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ ERROR: Missing columns: {missing_cols}")
        print(f"   Found columns: {list(df.columns)}")
        return
    
    # Extract data
    acc_data = df[['acc_x', 'acc_y', 'acc_z']].values
    ori_data = df[['ori_x', 'ori_y', 'ori_z', 'ori_w']].values
    
    print(f"   Acceleration shape: {acc_data.shape}")
    print(f"   Orientation shape: {ori_data.shape}")
    
    # Initialize processor
    print(f"\n🤖 Initializing MobilePoser (combo: {combo})...")
    try:
        processor = CustomIMUProcessor(model_path=model_path, combo=combo)
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ ERROR initializing model: {e}")
        return
    
    # Calibration
    print("\n📏 Calibrating (using first 90 frames)...")
    if len(acc_data) < 90:
        print(f"⚠️  WARNING: Only {len(acc_data)} frames available. Using all for calibration.")
        cal_frames = len(acc_data)
    else:
        cal_frames = 90
    
    try:
        processor.calibrate(acc_data[:cal_frames], ori_data[:cal_frames])
        print("   ✓ Calibration complete")
    except Exception as e:
        print(f"❌ ERROR during calibration: {e}")
        return
    
    # Process data
    print(f"\n🏃 Processing {len(acc_data) - cal_frames} frames...")
    try:
        poses, joints_list, translations = processor.predict_pose_sequence(
            acc_data[cal_frames:], 
            ori_data[cal_frames:]
        )
        print(f"   ✓ Processed {len(poses)} frames successfully")
    except Exception as e:
        print(f"❌ ERROR during inference: {e}")
        return
    
    # Save results
    output_file = Path(csv_file).stem + "_results.pt"
    print(f"\n💾 Saving results to: {output_file}")
    try:
        torch.save({
            'poses': torch.stack(poses),           # [N, 24, 3, 3] - SMPL pose parameters
            'joints': torch.stack(joints_list),    # [N, 24, 3] - 3D joint positions
            'translations': torch.stack(translations),  # [N, 3] - Root translations
            'combo': combo,
            'num_frames': len(poses)
        }, output_file)
        print("   ✓ Results saved")
    except Exception as e:
        print(f"❌ ERROR saving results: {e}")
        return
    
    # Print summary statistics
    print("\n📊 Summary Statistics:")
    translations_np = torch.stack(translations).cpu().numpy()
    print(f"   Translation range:")
    print(f"     X: [{translations_np[:, 0].min():.2f}, {translations_np[:, 0].max():.2f}] m")
    print(f"     Y: [{translations_np[:, 1].min():.2f}, {translations_np[:, 1].max():.2f}] m")
    print(f"     Z: [{translations_np[:, 2].min():.2f}, {translations_np[:, 2].max():.2f}] m")
    
    # Visualize if requested
    if visualize:
        print("\n📈 Generating visualization...")
        try:
            visualize_results(translations_np, joints_list)
        except Exception as e:
            print(f"⚠️  Visualization failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Test completed successfully!")
    print("=" * 60)


def test_with_synthetic_data(combo='rp', num_frames=100):
    """
    Test MobilePoser with synthetic (random) IMU data.
    Useful for testing if everything is set up correctly.
    
    Args:
        combo: Device combination (default: 'rp' for right pocket)
        num_frames: Number of frames to generate
    """
    print("=" * 60)
    print("MobilePoser Synthetic Data Test")
    print("=" * 60)
    
    # Check if weights file exists
    model_path = paths.weights_file
    if not Path(model_path).exists():
        print(f"\n❌ ERROR: Model weights not found at: {model_path}")
        print("\nPlease download the pretrained weights from:")
        print("https://uchicago.box.com/s/ey3y49srpo79propzvmjx0t8u3ael6cl")
        print(f"\nAnd place it at: {model_path}")
        return
    
    # Generate synthetic data
    print(f"\n🎲 Generating {num_frames} frames of synthetic IMU data...")
    np.random.seed(42)
    
    # Simulate standing still with small movements
    acc_data = np.random.randn(num_frames, 3) * 0.1
    acc_data[:, 1] += 9.8  # Add gravity in Y direction
    
    # Simulate small rotations around identity
    ori_data = np.tile([0, 0, 0, 1], (num_frames, 1))  # Identity quaternion
    ori_data[:, :3] += np.random.randn(num_frames, 3) * 0.01  # Small perturbations
    
    print("   ✓ Synthetic data generated")
    
    # Initialize processor
    print(f"\n🤖 Initializing MobilePoser (combo: {combo})...")
    try:
        processor = CustomIMUProcessor(model_path=model_path, combo=combo)
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ ERROR initializing model: {e}")
        return
    
    # Calibration
    print("\n📏 Calibrating (using first 90 frames)...")
    try:
        processor.calibrate(acc_data[:90], ori_data[:90])
        print("   ✓ Calibration complete")
    except Exception as e:
        print(f"❌ ERROR during calibration: {e}")
        return
    
    # Process data
    print(f"\n🏃 Processing {num_frames - 90} frames...")
    try:
        poses, joints_list, translations = processor.predict_pose_sequence(
            acc_data[90:], 
            ori_data[90:]
        )
        print(f"   ✓ Processed {len(poses)} frames successfully")
    except Exception as e:
        print(f"❌ ERROR during inference: {e}")
        return
    
    # Print summary
    print("\n📊 Summary:")
    print(f"   Output poses shape: {torch.stack(poses).shape}")
    print(f"   Output joints shape: {torch.stack(joints_list).shape}")
    print(f"   Output translations shape: {torch.stack(translations).shape}")
    
    print("\n" + "=" * 60)
    print("✅ Test completed successfully!")
    print("   Your setup is working correctly!")
    print("=" * 60)


def visualize_results(translations, joints_list):
    """Simple visualization of results."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("⚠️  Matplotlib not installed. Skipping visualization.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot translation trajectory
    axes[0].plot(translations[:, 0], translations[:, 2])
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Z (m)')
    axes[0].set_title('Root Translation Trajectory (Top View)')
    axes[0].grid(True)
    axes[0].axis('equal')
    
    # Plot translation over time
    axes[1].plot(translations[:, 0], label='X')
    axes[1].plot(translations[:, 1], label='Y')
    axes[1].plot(translations[:, 2], label='Z')
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Position (m)')
    axes[1].set_title('Root Translation Over Time')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('test_results_visualization.png', dpi=150)
    print("   ✓ Saved visualization to: test_results_visualization.png")
    plt.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="Simple test script for MobilePoser with custom data")
    parser.add_argument('--data', type=str, help='Path to CSV file with IMU data')
    parser.add_argument('--combo', type=str, default='rp',
                        help='Device combination (default: rp for right pocket)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model weights (default: use config path)')
    parser.add_argument('--synthetic', action='store_true',
                        help='Test with synthetic data (for setup verification)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--num-frames', type=int, default=100,
                        help='Number of synthetic frames to generate (default: 100)')
    
    args = parser.parse_args()
    
    if args.synthetic:
        # Test with synthetic data
        test_with_synthetic_data(combo=args.combo, num_frames=args.num_frames)
    elif args.data:
        # Test with real CSV data
        test_with_csv(args.data, combo=args.combo, model_path=args.model, 
                     visualize=args.visualize)
    else:
        print("Please specify either --data <csv_file> or --synthetic")
        parser.print_help()
