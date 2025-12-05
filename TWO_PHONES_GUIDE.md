# Using Two Phones for Better Pose Estimation

## Why Two Phones?

The MobilePoser model expects **5 IMU locations**:
- **Index 0**: Left wrist
- **Index 1**: Right wrist
- **Index 2**: Left thigh/pocket
- **Index 3**: Right thigh/pocket
- **Index 4**: Head

With only 2 IMUs (watch + 1 phone), the model has to **guess** 3 body parts, leading to:
- Poor leg tracking
- Weird sitting/curled poses
- Incorrect motion interpretation

**Adding a second phone dramatically improves results!**

## Recommended Setups

### Setup 1: Watch + Both Pockets (3 IMUs) ⭐ BEST
```bash
python mobileposer/another_example.py \
  --wristmotion-csv path/to/WristMotion.csv \
  --acc-csv path/to/phone1/Accelerometer.csv \
  --ori-csv path/to/phone1/Orientation.csv \
  --acc-csv2 path/to/phone2/Accelerometer.csv \
  --ori-csv2 path/to/phone2/Orientation.csv \
  --combo lw_lp_rp \
  --model mobileposer/checkpoints/weights.pth \
  --with-tran
```

**Placement:**
- Apple Watch: Left wrist
- Phone 1: Right pocket (index 3)
- Phone 2: Left pocket (index 2)

### Setup 2: Both Pockets Only (2 IMUs)
```bash
python mobileposer/another_example.py \
  --acc-csv path/to/phone1/Accelerometer.csv \
  --ori-csv path/to/phone1/Orientation.csv \
  --acc-csv2 path/to/phone2/Accelerometer.csv \
  --ori-csv2 path/to/phone2/Orientation.csv \
  --combo lp_rp \
  --model mobileposer/checkpoints/weights.pth \
  --with-tran
```

**Placement:**
- Phone 1: Right pocket (index 3)
- Phone 2: Left pocket (index 2)

### Setup 3: Watch + Right Pocket + Head (3 IMUs)
```bash
python mobileposer/another_example.py \
  --wristmotion-csv path/to/WristMotion.csv \
  --acc-csv path/to/phone1/Accelerometer.csv \
  --ori-csv path/to/phone1/Orientation.csv \
  --acc-csv2 path/to/phone2/Accelerometer.csv \
  --ori-csv2 path/to/phone2/Orientation.csv \
  --combo lw_rp_h \
  --model mobileposer/checkpoints/weights.pth \
  --with-tran
```

**Placement:**
- Apple Watch: Left wrist
- Phone 1: Right pocket (index 3)
- Phone 2: Head in headband/hat (index 4) - Need to modify code to place at index 4

## New Combos Available

I've added these new combinations to `config.py`:

| Combo | IMUs | Indices | Description |
|-------|------|---------|-------------|
| `lp_rp` | 2 | [2, 3] | Both pockets |
| `lw_lp_rp` | 3 | [0, 2, 3] | Left wrist + both pockets |
| `rw_lp_rp` | 3 | [1, 2, 3] | Right wrist + both pockets |

## Recording with Two Phones

### Synchronization
Both phones should start recording at approximately the same time. The code will:
1. Skip the first 3.5 seconds from both
2. Align timestamps
3. Trim to the shortest recording

### Tips
- Start both recordings within 1-2 seconds of each other
- Keep phones in pockets with screens facing your body
- Ensure phones don't rotate in pockets during movement
- Record the same calibration sequence on both phones

## Expected Improvements

With 2 phones (both pockets):
- ✅ Both legs tracked independently
- ✅ Better walking/running motion
- ✅ Correct turning and rotation
- ✅ No more weird sitting/curled poses

With 3 IMUs (watch + both pockets):
- ✅ Even better arm tracking
- ✅ More accurate upper body
- ✅ Better overall pose quality

## Current Limitation

The code currently places:
- Phone 1 → Index 3 (right pocket)
- Phone 2 → Index 2 (left pocket)

If you want phone 2 as head (index 4), you'll need to modify line 198-200 in `another_example.py`.

## Example Command

For your current setup with watch + phone in right pocket, if you add a second phone in left pocket:

```bash
python mobileposer/another_example.py \
  --wristmotion-csv ../../IMU-recordings/Ido-iPhone/test5-lw-lp-rp/WristMotion.csv \
  --acc-csv ../../IMU-recordings/Ido-iPhone/test5-lw-lp-rp/phone1/Accelerometer.csv \
  --ori-csv ../../IMU-recordings/Ido-iPhone/test5-lw-lp-rp/phone1/Orientation.csv \
  --acc-csv2 ../../IMU-recordings/Ido-iPhone/test5-lw-lp-rp/phone2/Accelerometer.csv \
  --ori-csv2 ../../IMU-recordings/Ido-iPhone/test5-lw-lp-rp/phone2/Orientation.csv \
  --combo lw_lp_rp \
  --model mobileposer/checkpoints/weights.pth \
  --with-tran
```

This should give you **much better results**!
