# Understanding Acceleration Data for MobilePoser

## The Problem: User Acceleration vs Total Acceleration

Apple's motion sensors (iPhone and Apple Watch) separate acceleration into **two components**:

1. **User Acceleration** - Movement-induced acceleration (without gravity)
2. **Gravity** - The gravity component in the sensor's reference frame

### Why This Matters

**MobilePoser expects TOTAL acceleration** (user acceleration + gravity) because:
- The model was trained on total acceleration data from AMASS dataset
- Gravity direction is crucial for determining sensor orientation
- Calibration relies on gravity to align coordinate frames with the body

## Your Data Files

### Apple Watch (WristMotion.csv)
✅ **Already has both components:**
- `accelerationX`, `accelerationY`, `accelerationZ` - User acceleration
- `gravityX`, `gravityY`, `gravityZ` - Gravity component
- `quaternionW`, `quaternionX`, `quaternionY`, `quaternionZ` - Orientation

**Solution:** Add them together: `total_acc = user_acc + gravity`

### iPhone (Accelerometer.csv + Orientation.csv)
❌ **Only has user acceleration:**
- Accelerometer.csv: `x`, `y`, `z` - User acceleration only
- Orientation.csv: `qw`, `qx`, `qy`, `qz` - Orientation

**Two Solutions:**

#### Option 1: Export Gravity Data (RECOMMENDED)
If your recording app can export gravity data, create a `Gravity.csv` file with columns:
```
time,x,y,z
```

Then use it:
```bash
python mobileposer/another_example.py \
  --acc-csv path/to/Accelerometer.csv \
  --ori-csv path/to/Orientation.csv \
  --gravity-csv path/to/Gravity.csv \
  --wristmotion-csv path/to/WristMotion.csv \
  --combo lw_rp
```

#### Option 2: Compute Gravity from Orientation (CURRENT)
If gravity data is not available, the code computes it from the orientation quaternion:
```python
# Rotate global gravity [0, 0, -1]g to sensor frame
sensor_gravity = rotation.apply([0, 0, -1])
total_acc = user_acc + sensor_gravity
```

This is less accurate because:
- Assumes perfect quaternion accuracy
- Doesn't account for sensor calibration differences
- May introduce small errors in gravity magnitude

## How to Record Proper Data

### For iPhone Apps
Use `CMMotionManager` and record **both**:
```swift
// User acceleration (without gravity)
let userAcceleration = motion.userAcceleration

// Gravity component
let gravity = motion.gravity

// Or use deviceMotion which provides both
if let motion = motionManager.deviceMotion {
    let userAcc = motion.userAcceleration
    let gravity = motion.gravity
    // Save both to CSV
}
```

### For Apple Watch Apps
Use `CMDeviceMotion` which provides:
```swift
let motion = motionManager.deviceMotion
let userAcc = motion.userAcceleration  // Without gravity
let gravity = motion.gravity            // Gravity component
let attitude = motion.attitude          // Orientation (convert to quaternion)
```

## Verification

After loading data, check the total acceleration magnitude:
```python
import numpy as np
total_acc_magnitude = np.linalg.norm(total_acc, axis=1).mean()
print(f"Total acceleration: {total_acc_magnitude:.3f}g")
```

**Expected values:**
- ✅ **~1.0g** - Correct! (gravity dominates when stationary)
- ❌ **~0.1-0.2g** - Wrong! Only user acceleration, missing gravity
- ❌ **~2.0g** - Wrong! Gravity might be added twice

## Current Implementation

The code now handles both cases:

1. **With gravity file** (preferred):
   ```python
   load_csv_data(acc_file, ori_file, gravity_file="Gravity.csv")
   ```

2. **Without gravity file** (fallback):
   ```python
   load_csv_data(acc_file, ori_file)  # Computes gravity from orientation
   ```

## Summary

| Data Source | Has Gravity? | Solution |
|-------------|--------------|----------|
| WristMotion.csv | ✅ Yes | Use `gravity + userAcceleration` |
| iPhone Accelerometer.csv | ❌ No | Export gravity OR compute from orientation |

**Recommendation:** Update your iPhone recording app to export gravity data separately for best accuracy.
