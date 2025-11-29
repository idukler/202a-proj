"""
Modified live_demo.py for phone IMU data
Changes: Replace IMUSet with PhoneIMUBuffer, remove calibration
"""

import os
import time
import socket
import threading
import torch
import numpy as np
from argparse import ArgumentParser
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from pygame.time import Clock

from articulate.math import *
from mobileposer.models import *
from mobileposer.utils.model_utils import *
from mobileposer.config import *

# Import phone receiver
from phone_imu_receiver import phone_buffer, start_server

def get_input():
    global running
    while running:
        c = input()
        if c == 'q':
            running = False

def visualize_pose_pygame(pose, tran, screen, font):
    """
    Visualize pose using PyGame - same logic as Unity visualization
    Receives pose (72 axis-angle) and translation (3) and displays them
    
    Args:
        pose: [72] axis-angle representation of 24 joints
        tran: [3] global translation
        screen: pygame screen object
        font: pygame font object
    """
    # SMPL skeleton hierarchy (parent joint indices)
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    
    # Joint names for reference
    joint_names = [
        'Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 'L_Ankle', 'R_Ankle',
        'Spine3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar', 'R_Collar', 'Head',
        'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand'
    ]
    
    # SMPL bone lengths (approximate, in meters)
    bone_lengths = [
        0.0, 0.1, 0.1, 0.1, 0.4, 0.4, 0.1, 0.4, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15,
        0.15, 0.15, 0.3, 0.3, 0.25, 0.25, 0.1, 0.1
    ]
    
    # Convert pose from axis-angle to rotation matrices for FK
    pose_aa = pose.view(24, 3)
    
    # Compute forward kinematics to get joint positions
    joint_positions = torch.zeros(24, 3)
    joint_rotations = []
    
    for i in range(24):
        # Get rotation for this joint
        aa = pose_aa[i]
        angle = torch.norm(aa)
        if angle > 1e-8:
            axis = aa / angle
            # Rodrigues formula to convert axis-angle to rotation matrix
            K = torch.tensor([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R = torch.eye(3) + torch.sin(angle) * K + (1 - torch.cos(angle)) * K @ K
        else:
            R = torch.eye(3)
        joint_rotations.append(R)
        
        # Compute position using parent's position and rotation
        if parents[i] == -1:
            joint_positions[i] = tran
        else:
            parent_idx = parents[i]
            # Direction from parent to child (simplified - along Y axis)
            offset = torch.tensor([0.0, bone_lengths[i], 0.0])
            # Rotate offset by parent's rotation
            rotated_offset = joint_rotations[parent_idx] @ offset
            joint_positions[i] = joint_positions[parent_idx] + rotated_offset
    
    # Clear screen
    screen.fill((20, 20, 30))
    
    # Projection settings
    screen_w, screen_h = screen.get_size()
    scale = 150  # pixels per meter
    center_x = screen_w // 2
    center_y = screen_h - 100  # Bottom offset
    
    def project(pos):
        """Project 3D position to 2D screen (front view)"""
        x = int(center_x + pos[0].item() * scale)
        y = int(center_y - pos[1].item() * scale)  # Flip Y
        return (x, y)
    
    # Draw bones (connections)
    for i in range(24):
        if parents[i] != -1:
            p1 = project(joint_positions[parents[i]])
            p2 = project(joint_positions[i])
            
            # Color code: blue for left, red for right, white for center
            if 'L_' in joint_names[i]:
                color = (100, 150, 255)
            elif 'R_' in joint_names[i]:
                color = (255, 100, 100)
            else:
                color = (200, 200, 200)
            
            pygame.draw.line(screen, color, p1, p2, 3)
    
    # Draw joints
    for i in range(24):
        pos = project(joint_positions[i])
        if i == 0:  # Pelvis
            color = (255, 255, 0)
            radius = 8
        elif i == 15:  # Head
            color = (255, 200, 100)
            radius = 10
        else:
            color = (150, 255, 150)
            radius = 5
        
        pygame.draw.circle(screen, color, pos, radius)
        pygame.draw.circle(screen, (255, 255, 255), pos, radius, 1)
    
    # Display info
    fps_text = font.render(f'FPS: {pygame.time.Clock().get_fps():.1f}', True, (0, 255, 0))
    screen.blit(fps_text, (10, 10))
    
    tran_text = font.render(f'Translation: ({tran[0]:.2f}, {tran[1]:.2f}, {tran[2]:.2f})', True, (200, 200, 200))
    screen.blit(tran_text, (10, 40))
    
    info_text = font.render('Press Q to quit', True, (150, 150, 150))
    screen.blit(info_text, (10, screen_h - 30))
    
    pygame.display.flip()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--save", action='store_true')
    args = parser.parse_args()
    
    # Start HTTP server in background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    print("HTTP server started on http://0.0.0.0:8000/data")
    print("Waiting for phone data...")
    time.sleep(2)
    
    raise ValuerError("done")

    
    # Specify device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = load_model(paths.weights_file)
    
    raise ValuerError(model.params)
    
    # Setup PyGame visualization
    if args.vis:
        pygame.init()
        screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption('MobilePoser - Real-time Pose Visualization')
        font = pygame.font.Font(None, 30)
        print('PyGame visualization initialized.')
    
    running = True
    clock = Clock()
    
    get_input_thread = threading.Thread(target=get_input, daemon=True)
    get_input_thread.start()
    
    n_imus = 5
    poses, trans = [], []
    
    # Simple identity transformation (no calibration needed for testing)
    smpl2imu = torch.eye(3)
    device2bone = torch.eye(3).unsqueeze(0).repeat(n_imus, 1, 1)
    acc_offsets = torch.zeros(n_imus, 3, 1)
    
    model.eval()
    print("Ready. Press 'q' to quit")
    
    while running:
        clock.tick(datasets.fps)
        
        # Get data from phone buffer
        ori_raw, acc_raw = phone_buffer.get_current_buffer()
        
        if ori_raw.shape[0] == 0:
            time.sleep(0.01)
            continue
        
        # Convert quaternions to rotation matrices
        ori_raw = quaternion_to_rotation_matrix(ori_raw).view(-1, n_imus, 3, 3)
        
        # Apply transformations (simplified - using identity)
        glb_acc = (smpl2imu.matmul(acc_raw.view(-1, n_imus, 3, 1)) - acc_offsets).view(-1, n_imus, 3)
        glb_ori = smpl2imu.matmul(ori_raw).matmul(device2bone)
        
        # Normalization (reorder for combo: lw_rp)
        _acc = glb_acc.view(-1, 5, 3)[:, [1, 4, 3, 0, 2]] / amass.acc_scale
        _ori = glb_ori.view(-1, 5, 3, 3)[:, [1, 4, 3, 0, 2]]
        
        acc = torch.zeros_like(_acc)
        ori = torch.zeros_like(_ori)
        
        # Device combo (using phone as multiple sensors)
        combo = 'rp'
        c = amass.combos[combo]
        acc[:, c] = _acc[:, c]
        ori[:, c] = _ori[:, c]
        
        imu_input = torch.cat([acc.flatten(1), ori.flatten(1)], dim=1)
        
        # Predict pose and translation
        with torch.no_grad():
            output = model.forward_online(imu_input.squeeze(0), [imu_input.shape[0]])
            pred_pose = output[0]  # [24, 3, 3]
            pred_tran = output[2]  # [3]
        
        # Convert to axis angle
        pose = rotation_matrix_to_axis_angle(pred_pose.view(1, 216)).view(72)
        tran = pred_tran
        
        # Save for later
        if args.save:
            poses.append(pred_pose)
            trans.append(pred_tran)
        
        # Visualize with PyGame
        if args.vis:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            visualize_pose_pygame(pose, tran, screen, font)
        
        if os.getenv("DEBUG"):
            print(f'\rOutput FPS: {clock.get_fps():.1f}', end='')
    
    # Save data
    if args.save and poses:
        data = {
            'pose': torch.stack(poses, dim=0),
            'tran': torch.stack(trans, dim=0),
        }
        torch.save(data, paths.dev_data / f'phone_data_{int(time.time())}.pt')
    
    print('\nFinish.')