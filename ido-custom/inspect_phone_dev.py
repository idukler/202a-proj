import sys
from pathlib import Path

import torch


def main(path_str: str):
    path = Path(path_str)
    if not path.exists():
        print(f"File not found: {path}")
        return

    data = torch.load(path, map_location="cpu")

    print("Loaded keys:", list(data.keys()))

    def show_tensor(name):
        if name not in data:
            print(f"{name}: <missing>")
            return
        t = data[name]
        if not torch.is_tensor(t):
            print(f"{name}: type={type(t)} value={t}")
            return
        print(f"{name}: shape={tuple(t.shape)}, dtype={t.dtype}")
        if t.numel() > 0:
            flat = t.reshape(-1)
            print(f"  first 5 values: {flat[:5].tolist()}")

    # Main tensors
    for name in ["raw_acc", "raw_ori", "acc", "ori", "pose", "tran"]:
        show_tensor(name)

    # Calibration block
    calib = data.get("calibration", None)
    if calib is None:
        print("calibration: <missing>")
    else:
        print("calibration keys:", list(calib.keys()))
        for name in ["smpl2imu", "device2bone"]:
            if name in calib:
                t = calib[name]
                print(f"  {name}: shape={tuple(t.shape)}, dtype={t.dtype}")
                flat = t.reshape(-1)
                print(f"    first 5 values: {flat[:5].tolist()}")
            else:
                print(f"  {name}: <missing>")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_phone_dev.py path/to/phone_dev_xxx.pt")
    else:
        main(sys.argv[1])
