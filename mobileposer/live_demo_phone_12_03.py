"""
Code adapted from: https://github.com/Xinyu-Yi/TransPose/blob/main/live_demo.py
Modified to use HTTP instead of UDP for phone IMU data
Modified to use PyGame instead of Unity for visualization
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
from pygame_visualizer import PoseVisualizer
import pickle
from flask import Flask, request
from threading import Lock

from articulate.math import *
from mobileposer.models import *
from mobileposer.utils.model_utils import *
from mobileposer.config import *

# Configurations 
USE_PHONE_AS_WATCH = False


class PhoneIMUSet:
    """
    HTTP-based IMU receiver for phone data.
    Mimics the original IMUSet interface but receives data via HTTP instead of UDP.
    """
    def __init__(self, imu_host='0.0.0.0', imu_port=8000, buffer_len=26):
        """
        Init a PhoneIMUSet for receiving phone IMU data via HTTP.

        :param imu_host: The host to bind the HTTP server to.
        :param imu_port: The port to bind the HTTP server to.
        :param buffer_len: Max number of frames in the readonly buffer.
        """
        self.imu_host = imu_host
        self.imu_port = imu_port
        self.clock = Clock()

        self._buffer_len = buffer_len
        self._quat_buffer = []
        self._acc_buffer = []
        self._is_reading = False
        self._read_thread = None
        self._lock = Lock()
        
        # Flask app for HTTP server
        self._app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self):
        """Setup Flask routes"""
        @self._app.route('/data', methods=['POST'])
        def receive_data():
            """Endpoint to receive phone IMU data"""
            try:
                data = request.get_json()
                
                # Extract orientation and acceleration from payload
                quat = None
                acc = None
                
                if 'payload' in data and len(data['payload']) > 0:
                    # Find orientation and accelerometer data
                    for item in reversed(data['payload']):
                        if item['name'] == 'orientation' and quat is None:
                            values = item['values']
                            quat = np.array([
                                values['qw'],
                                values['qx'],
                                values['qy'],
                                values['qz'],
                               # was here values['qw']
                            ])
                        elif item['name'] == 'accelerometer' and acc is None:
                            values = item['values']
                            acc = np.array([
                                values['x'],
                                values['y'],
                                values['z']
                            ])
                        
                        if quat is not None and acc is not None:
                            break
                    
                    if quat is not None and acc is not None:
                        # Format data to match original IMUSet format
                        # Replicate phone data to all 5 IMU positions
                        quat_full = np.tile(quat, (5, 1))  # [5, 4]
                        acc_full = np.tile(acc, (5, 1)) * -9.8  # [5, 3]
                        
                        with self._lock:
                            tranc = int(len(self._quat_buffer) == self._buffer_len)
                            self._quat_buffer = self._quat_buffer[tranc:] + [quat_full.astype(float)]
                            self._acc_buffer  = self._acc_buffer[tranc:] + [acc_full.astype(float)]
                            self.clock.tick()
                
                return {'status': 'ok'}, 200
            except Exception as e:
                print(f"Error processing data: {e}")
                return {'status': 'error', 'message': str(e)}, 400

    def _run_server(self):
        """Run Flask server in thread"""
        self._app.run(host=self.imu_host, port=self.imu_port, debug=False, threaded=True)

    def start_reading(self):
        """
        Start reading imu measurements into the buffer.
        """
        if self._read_thread is None:
            self._is_reading = True
            self._read_thread = threading.Thread(target=self._run_server)
            self._read_thread.setDaemon(True)
            self._quat_buffer = []
            self._acc_buffer = []
            self._read_thread.start()
            print(f"HTTP server started on http://{self.imu_host}:{self.imu_port}/data")
        else:
            print('Failed to start reading thread: reading is already start.')

    def stop_reading(self):
        """
        Stop reading imu measurements.
        """
        if self._read_thread is not None:
            self._is_reading = False
            # Note: Flask doesn't have a clean shutdown in thread mode
            # The daemon thread will be killed when main thread exits
            self._read_thread = None

    def get_current_buffer(self):
        """
        Get a view of current buffer.

        :return: Quaternion and acceleration torch.Tensor in shape [buffer_len, 5, 4] and [buffer_len, 5, 3].
        """
        with self._lock:
            q = torch.from_numpy(np.array(self._quat_buffer)).float()
            a = torch.from_numpy(np.array(self._acc_buffer)).float()
            return q, a

    def get_mean_measurement_of_n_second(self, num_seconds=3, buffer_len=120):
        """
        Collect data for `num_seconds` seconds. The average of the collected
        frames of the measured quaternions and accelerations are returned.
        Note that this function is blocking.

        :param num_seconds: How many seconds to collect data.
        :param buffer_len: Buffer length. Must be smaller than 60 * `num_seconds`.
        :return: The mean quaternion and acceleration torch.Tensor in shape [5, 4] and [5, 3] respectively.
        """
        save_buffer_len = self._buffer_len
        self._buffer_len = buffer_len
        # Clear buffer and collect fresh data
        with self._lock:
            self._quat_buffer = []
            self._acc_buffer = []
        # If not already reading, start
        if self._read_thread is None:
            self.start_reading()
        time.sleep(num_seconds)
        q, a = self.get_current_buffer()
        self._buffer_len = save_buffer_len
        return q.mean(dim=0), a.mean(dim=0)


def get_input():
    global running, start_recording
    while running:
        c = input()
        if c == 'q':
            running = False
        elif c == 'r':
            start_recording = True
        elif c == 's':
            start_recording = False


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--save", action='store_true')
    args = parser.parse_args()
    
    # specify device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # setup IMU collection
    imu_set = PhoneIMUSet(buffer_len=1)

    # Wait for first packet before calibration
    imu_set.start_reading()
    print('Waiting for phone to send data...')
    while len(imu_set._quat_buffer) == 0:
        time.sleep(0.1)
    print('Receiving data from phone!')

    # align IMU to SMPL body frame
    input('Put imu 1 aligned with your body reference frame (x = Left, y = Up, z = Forward) and then press any key.')
    print('Capturing data for 3 seconds...')
    time.sleep(2)  # Brief wait for buffer to start filling
    if len(imu_set._quat_buffer) == 0:
        print('ERROR: No data received. Check phone connection.')
        exit(1)
    oris = imu_set.get_mean_measurement_of_n_second(num_seconds=3, buffer_len=40)[0][0]    
    smpl2imu = quaternion_to_rotation_matrix(oris).view(3, 3).t()
    input('Data collected! Press any key to continue to T-pose calibration.')
    

    # bone and acceleration offset calibration
    input('\tFinish.\nKeep phone in right pocket and press any key.')
    for i in range(3, 0, -1):
        print('\rStand straight in T-pose and be ready. The calibration will begin after %d seconds.' % i, end='')
        time.sleep(1)
    print('\nStand straight in T-pose. Keep the pose for 3 seconds ...', end='')

    oris, accs = imu_set.get_mean_measurement_of_n_second(num_seconds=3, buffer_len=40)
    oris = quaternion_to_rotation_matrix(oris)
    device2bone = smpl2imu.matmul(oris).transpose(1, 2).matmul(torch.eye(3))
    acc_offsets = smpl2imu.matmul(accs.unsqueeze(-1))   # [num_imus, 3, 1], already in global inertial frame

    # start streaming data
    print('\tFinished Calibrating.\nEstimating poses. Press q to quit')

    # load model
    model = load_model(paths.weights_file)
    
    # setup PyGame visualization
    if args.vis:
        visualizer = PoseVisualizer()

    running = True
    clock = Clock()
    is_recording = False
    record_buffer = None

    get_input_thread = threading.Thread(target=get_input)
    get_input_thread.setDaemon(True)
    get_input_thread.start()

    n_imus = 5
    accs, oris = [], []
    raw_accs, raw_oris = [], []
    poses, trans = [], []

    model.eval()
    while running:
        # calibration
        clock.tick(datasets.fps)
        ori_raw, acc_raw = imu_set.get_current_buffer() # [buffer_len, 5, 4]
        
        if ori_raw.shape[0] == 0:
            time.sleep(0.01)
            continue
        
        # Take only the last frame
        ori_raw = ori_raw[-1:]
        print(ori_raw.shape)
        acc_raw = acc_raw[-1:]
        
        ori_raw = quaternion_to_rotation_matrix(ori_raw).view(-1, n_imus, 3, 3)
        glb_acc = (smpl2imu.matmul(acc_raw.view(-1, n_imus, 3, 1)) - acc_offsets).view(-1, n_imus, 3)
        glb_ori = smpl2imu.matmul(ori_raw).matmul(device2bone)
        
        # ADDITION
        flip = torch.diag(torch.tensor([1.0, -1.0, 1.0], device=glb_ori.device))
        glb_ori = glb_ori.matmul(flip)

        # normalization 
        _acc = glb_acc.view(-1, 5, 3)[:, [1, 4, 3, 0, 2]] / amass.acc_scale
        _ori = glb_ori.view(-1, 5, 3, 3)[:, [1, 4, 3, 0, 2]]
        acc = torch.zeros_like(_acc)
        ori = torch.zeros_like(_ori)


        # device combo - using only right pocket
        combo = 'rp'
        c = amass.combos[combo]

        if USE_PHONE_AS_WATCH:
            # set watch value to phone
            acc[:, [0]] = _acc[:, [3]]
            ori[:, [0]] = _ori[:, [3]]
        else:
            # filter and concat input
            acc[:, c] = _acc[:, c] 
            ori[:, c] = _ori[:, c]
        
        imu_input = torch.cat([acc.flatten(1), ori.flatten(1)], dim=1)

        # predict pose and translation
        with torch.no_grad():
            output = model.forward_online(imu_input.squeeze(0), [imu_input.shape[0]])
            pred_pose = output[0] # [24, 3, 3]
            pred_tran = output[2] # [3]
        
        # convert rotmatrix to axis angle
        pose = rotation_matrix_to_axis_angle(pred_pose.view(1, 216)).view(72)
        tran = pred_tran

        # keep track of data
        if args.save:
            accs.append(glb_acc)
            oris.append(glb_ori)
            raw_accs.append(acc_raw)
            raw_oris.append(ori_raw)
            poses.append(pred_pose)
            trans.append(pred_tran)

        # visualize with PyGame
        if args.vis:
            if not visualizer.handle_events():
                running = False
            visualizer.draw_pose(pose, tran)
            
            if os.getenv("DEBUG") is not None:
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
            'calibration': {
                'smpl2imu': smpl2imu,
                'device2bone': device2bone
            }
        }
        torch.save(data, f'phone_dev_{int(time.time())}.pt')

    # clean up
    if args.vis:
        visualizer.close()
    get_input_thread.join()
    imu_set.stop_reading()
    print('Finish.')