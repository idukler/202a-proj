"""
HTTP receiver for phone IMU data - minimal addition to existing system
Receives JSON data and converts to format expected by live_demo.py
"""

import json
import numpy as np
import torch
from flask import Flask, request
from threading import Lock

app = Flask(__name__)

class PhoneIMUBuffer:
    def __init__(self, buffer_len=1):
        self.buffer_len = buffer_len
        self.quat_buffer = []
        self.acc_buffer = []
        self.lock = Lock()
        
    def update(self, quat, acc):
        """Update buffers with new data from phone
        Args:
            quat: [4] quaternion [qx, qy, qz, qw]
            acc: [3] acceleration [x, y, z]
        """
        with self.lock:
            # Maintain buffer size
            if len(self.quat_buffer) >= self.buffer_len:
                self.quat_buffer.pop(0)
                self.acc_buffer.pop(0)
            
            self.quat_buffer.append(quat)
            self.acc_buffer.append(acc)
    
    def get_current_buffer(self):
        """Get buffer in format expected by live_demo.py
        Returns:
            quat: torch.Tensor [buffer_len, n_imus, 4]
            acc: torch.Tensor [buffer_len, n_imus, 3]
        """
        with self.lock:
            if not self.quat_buffer:
                # Return empty buffers if no data
                return (torch.zeros(0, 5, 4).float(), 
                        torch.zeros(0, 5, 3).float())
            
            # Convert to numpy arrays
            q_array = np.array(self.quat_buffer)  # [buffer_len, 4]
            a_array = np.array(self.acc_buffer)   # [buffer_len, 3]
            
            # Expand to 5 IMUs (replicate phone data to all positions)
            q_expanded = np.repeat(q_array[:, np.newaxis, :], 5, axis=1)  # [buffer_len, 5, 4]
            a_expanded = np.repeat(a_array[:, np.newaxis, :], 5, axis=1)  # [buffer_len, 5, 3]
            
            return (torch.from_numpy(q_expanded).float(),
                    torch.from_numpy(a_expanded).float())

# Global buffer instance
phone_buffer = PhoneIMUBuffer(buffer_len=1)

@app.route('/data', methods=['POST'])
def receive_data():
    """Endpoint to receive phone IMU data"""
    try:
        data = request.get_json()
        
        # Extract latest sensor readings from payload
        if 'payload' in data and len(data['payload']) > 0:
            latest = data['payload'][-1]  # Get most recent reading
            
            # Extract orientation (quaternion)
            if 'name' in latest and latest['name'] == 'orientation':
                values = latest['values']
                quat = np.array([
                    values['qx'],
                    values['qy'], 
                    values['qz'],
                    values['qw']
                ])
            else:
                print("No orientation data found")
                # Find orientation in payload
                for item in reversed(data['payload']):
                    if item['name'] == 'orientation':
                        values = item['values']
                        quat = np.array([
                            values['qx'],
                            values['qy'],
                            values['qz'],
                            values['qw']
                        ])
                        break
            
            # Extract accelerometer
            for item in reversed(data['payload']):
                if item['name'] == 'accelerometer':
                    values = item['values']
                    acc = np.array([
                        values['x'],
                        values['y'],
                        values['z']
                    ])
                    break
            
            # Update buffer
            phone_buffer.update(quat, acc)
            
        return {'status': 'ok'}, 200
    except Exception as e:
        print(f"Error processing data: {e}")
        return {'status': 'error', 'message': str(e)}, 400

def start_server(host='0.0.0.0', port=8000):
    """Start Flask server"""
    app.run(host=host, port=port, debug=False, threaded=True)

if __name__ == '__main__':
    start_server()