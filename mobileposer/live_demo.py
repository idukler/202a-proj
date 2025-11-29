"""
Code adapted from: https://github.com/Xinyu-Yi/TransPose/blob/main/live_demo.py
"""

import os
import time
import socket
import threading
import torch
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
from pygame.time import Clock
import pickle

# Removed the direct import from 'articulate.math' as it was causing a ModuleNotFoundError.
# Assuming necessary functions like rotation_matrix_to_axis_angle are available 
# via the mobileposer.utils.model_utils import below.

# FIX: Use absolute imports instead of relative imports
from mobileposer.models import *
from mobileposer.utils.model_utils import * from mobileposer.config import *
from mobileposer.file_stream_adapter import FileIMUSet 

# Configurations 
USE_PHONE_AS_WATCH = False


class IMUSet:
# ... (IMUSet class implementation remains unchanged) ...

    def close(self):
        try:
            self.conn.close()
        except:
            pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7777)
    parser.add_argument('--imu-count', type=int, default=6)
    parser.add_argument('--vis', action='store_true', help='Visualize in real-time (requires separate viewer).')
    parser.add_argument('--save', action='store_true', help='Save the raw and processed data.')
    # New arguments for file streaming
    parser.add_argument('--acc-file', type=str, default=None, help='Path to Accelerometer CSV for file streaming.')
    parser.add_argument('--ori-file', type=str, default=None, help='Path to Orientation CSV for file streaming.')
    # Added argument for IMU combo
    parser.add_argument('--combo', type=str, default='all', help='IMU combination being used (e.g., lw_rp, rp, all).') 
    
    args = parser.parse_args()

    # Determine if running file stream or live socket
    is_file_stream = args.acc_file and args.ori_file
    
    if is_file_stream:
        print(f"Running in file streaming mode: {args.acc_file} and {args.ori_file}")
        # Use the FileIMUSet adapter
        imu_set = FileIMUSet(acc_path=args.acc_file, ori_path=args.ori_file)
    else:
        # Original socket mode
        imu_set = IMUSet(imu_host=args.host, imu_port=args.port)

    # load model
    # Use the combo argument provided by the user (defaulting to 'all')
    model = Poser(num_imu=6, combo=args.combo)
    try:
        # Use args.model path which defaults to paths.weights_file if not provided
        # Ensure model is on CPU if no CUDA is available, although the main loop forces CUDA
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(paths.weights_file, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model weights not found at {paths.weights_file}")
        print("Please train the model or download pre-trained weights.")
        exit()
        
    # Move model to CUDA if available, otherwise it stays on CPU from map_location above.
    model.to(device)
    model.eval()
    
    # buffers
    buffer_len = 26
    input_buffer = []
    
    # calibration
    # Move calibration tensors to the chosen device (cuda or cpu)
    acc_calib, ori_calib = imu_set.calibrate()
    acc_calib = acc_calib.to(device)
    ori_calib = ori_calib.to(device)
    
    # connection to viewer
    if args.vis:
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        conn.bind(('127.0.0.1', 8888))
        conn.listen(1)
        print("Waiting for viewer to connect on port 8888...")
        conn, addr = conn.accept()
        print('Viewer connected by', addr)
    
    # misc
    clock = Clock()
    accs, oris, raw_accs, raw_oris, poses, trans = [], [], [], [], [], []
    
    is_recording = True
    while is_recording:
        # get imu data
        acc_raw, ori_raw, glb_acc, glb_ori, is_recording = imu_set.get_imu_data()
        
        # Stop if file reading is finished or original stream is terminated
        if not is_recording:
            break

        # apply calibration and transform raw data to global
        # Move raw data to device before transformation
        glb_acc, glb_ori = imu_set.transform(acc_raw.to(device), ori_raw.to(device))
        
        # prepare model input
        # Data is already on device from transform step
        input_data = torch.cat([glb_acc, glb_ori], dim=-1).flatten(1)
        
        # update input buffer
        input_buffer.append(input_data)
        if len(input_buffer) < buffer_len:
            continue
            
        # inference
        with torch.no_grad():
            # Concatenate input buffer tensors and run model
            pred_pose, pred_tran, pred_vel = model(torch.cat(input_buffer, dim=0).unsqueeze(0))

        # remove oldest data
        input_buffer.pop(0)

        # convert rotmatrix to axis angle
        # Move back to CPU for numpy/socket operations
        pose = rotation_matrix_to_axis_angle(pred_pose.view(1, 216)).view(72).cpu().numpy().flatten()
        tran = pred_tran.cpu().numpy().flatten()

        # keep track of data
        if args.save:
            # Store data on CPU to avoid filling VRAM
            accs.append(glb_acc.cpu())
            oris.append(glb_ori.cpu())
            raw_accs.append(acc_raw.cpu())
            raw_oris.append(ori_raw.cpu())
            poses.append(pred_pose.cpu())
            trans.append(pred_tran.cpu())

        # send pose
        if args.vis:
            s = ','.join(['%g' % v for v in pose]) + '#' + \
                ','.join(['%g' % v for v in tran]) + '$'
            conn.send(s.encode('utf8'))  
            
            if os.getenv("DEBUG") is not None:
                # Switched imu_set.clock.get_fps() to imu_set.clock.get_fps() for adapter compatibility
                print('\r', '(recording)' if is_recording else '', 'Sensor FPS:', imu_set.clock.get_fps(),
                        '\tOutput FPS:', clock.get_fps(), end='')

    # save data to file for viewer
    if args.save:
        data = {
            'raw_acc': torch.cat(raw_accs, dim=0),
            'raw_ori': torch.cat(raw_oris, dim=0),
            'acc': torch.cat(accs, dim=0),
            'ori': torch.cat(oris, dim=0),
            'pose': torch.cat(poses, dim=0),
            'tran': torch.cat(trans, dim=0),
            'calibration': imu_set.calibration
        }
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        save_path = paths.eval_dir / f"live_output_{timestamp}.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"\nData saved to {save_path}")

    imu_set.close()
    if args.vis:
        conn.close()