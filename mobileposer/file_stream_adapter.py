import pandas as pd
import numpy as np
import torch
import time
from typing import Tuple, List

# Define the IMU ID for the single sensor data (assuming Pelvis/Root)
# Based on IMUSet docstring: l_forearm (0), r_forearm (1), l_leg (2), r_leg (3), head (4), pelvis (5)
PELVIS_IMU_ID = 5 

class FileIMUSet:
    """
    Simulates the IMUSet streaming data by loading pre-recorded data 
    from Acceleration.csv and Orientation.csv and iterating over it 
    with a time delay.
    
    The single sensor data is placed into the 'Pelvis' (ID 5) slot.
    """
    def __init__(self, acc_path: str='Accelerometer.csv', ori_path: str='Orientation.csv'):
        # 1. Load DataFrames
        try:
            # Using the exact uploaded filenames
            acc_df = pd.read_csv(acc_path)
            ori_df = pd.read_csv(ori_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find CSV file: {e.filename}. Ensure files are in the correct path.")

        # 2. Select and reorder columns
        # Acceleration: [x, y, z]. CSV columns are (z, y, x). We reorder for standard XYZ.
        acc_data = acc_df[['x', 'y', 'z']] 
        
        # Orientation (Quaternion): [qw, qx, qy, qz]. CSV columns are (qx, qz, qw, qy).
        ori_data = ori_df[['qw', 'qx', 'qy', 'qz']]

        # 3. Merge data on timestamp (time is in nanoseconds/microseconds)
        # We merge based on timestamps to combine acceleration and orientation readings.
        merged_df = pd.merge(
            acc_df[['time', 'seconds_elapsed']],
            acc_data,
            on='time',
            how='outer'
        )
        merged_df = pd.merge(
            merged_df,
            ori_data,
            on='time',
            how='outer'
        )

        # 4. Interpolate missing data to synchronize frames
        # Use linear interpolation to fill gaps where one sensor was sampled but the other wasn't
        merged_df = merged_df.sort_values(by='time').interpolate(method='linear')
        merged_df = merged_df.dropna() # Drop any remaining NaNs (e.g., first/last row gaps)

        # 5. Convert to PyTorch tensors and create iterator
        self.acc_data = torch.from_numpy(merged_df[['x', 'y', 'z']].values).float()
        self.ori_data = torch.from_numpy(merged_df[['qw', 'qx', 'qy', 'qz']].values).float()
        self.timestamps = merged_df['time'].values
        self.seconds_elapsed = merged_df['seconds_elapsed'].values
        
        self.iterator = iter(zip(self.acc_data, self.ori_data, self.timestamps, self.seconds_elapsed))
        self.last_elapsed_time = self.seconds_elapsed[0]
        self.frame_count = 0
        self.start_time = time.time()

        # Placeholders for required outputs (1, 6, 3) acc and (1, 6, 4) ori
        self.acc_raw = torch.zeros(1, 6, 3)
        self.ori_raw = torch.zeros(1, 6, 4)
        self.glb_acc = torch.zeros(1, 6, 3)
        self.glb_ori = torch.zeros(1, 6, 4)
        self.calibration = None # Required by live_demo.py for saving

    def calibrate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mimics the calibration step but returns default zeros for simplicity."""
        print("--- Simulating Calibration... (Using default zero calibration) ---")
        self.calibration = {
            'acc_calib': torch.zeros(6, 3), 
            'ori_calib': torch.tensor([1., 0., 0., 0.]).repeat(6, 1) # Identity quaternion
        }
        return self.calibration['acc_calib'], self.calibration['ori_calib']

    def transform(self, acc_raw: torch.Tensor, ori_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mimics the transformation step. Since we lack the specific sensor-to-global 
        transformation, we assume the data from file (Pelvis slot) is already 
        in a usable 'global' frame for simulation.
        """
        self.glb_acc = acc_raw.clone() 
        self.glb_ori = ori_raw.clone()

        return self.glb_acc, self.glb_ori

    def get_imu_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """
        Returns a single frame of IMU data and simulates the streaming delay.
        
        Returns: 
            acc_raw (1, 6, 3), ori_raw (1, 6, 4), glb_acc (1, 6, 3), glb_ori (1, 6, 4), is_recording (bool)
        """
        try:
            # Get next frame
            acc_frame, ori_frame, timestamp, elapsed_time = next(self.iterator)
        except StopIteration:
            # End of file, return None to stop the main loop
            return None, None, None, None, False 

        # Simulate streaming delay based on the time difference between frames
        time_diff_sec = elapsed_time - self.last_elapsed_time
        # Sleep for the time difference, compensating for execution time
        time.sleep(max(0, time_diff_sec - (time.time() - self.start_time - elapsed_time)))
        self.last_elapsed_time = elapsed_time
        
        # Update raw data tensors, filling only the Pelvis IMU slot (ID 5)
        self.acc_raw.zero_()
        self.ori_raw.zero_()
        self.acc_raw[0, PELVIS_IMU_ID] = acc_frame # Set acc for Pelvis IMU
        self.ori_raw[0, PELVIS_IMU_ID] = ori_frame # Set ori for Pelvis IMU

        # Reset global data, transform() will fill it in the live_demo loop
        self.glb_acc.zero_()
        self.glb_ori.zero_()
        
        self.frame_count += 1
        # is_recording is True as long as there is data to stream
        return self.acc_raw.clone(), self.ori_raw.clone(), self.glb_acc.clone(), self.glb_ori.clone(), True

    def reset_clock(self):
        """Resets the streaming clock."""
        self.start_time = time.time()

    @property
    def clock(self):
        """Mock clock for compatibility with live_demo.py FPS display."""
        class MockClock:
            def __init__(self, parent):
                self.parent = parent
            def get_fps(self):
                if self.parent.frame_count > 0:
                    return self.parent.frame_count / (time.time() - self.parent.start_time)
                return 0.0
            def tick(self, fps): pass
            def get_time(self): return (time.time() - self.parent.start_time) * 1000
            def get_raw_time(self): return self.get_time()

        return MockClock(parent=self)
        
    def close(self):
        """Clean up."""
        print("\n--- File streaming finished. ---")