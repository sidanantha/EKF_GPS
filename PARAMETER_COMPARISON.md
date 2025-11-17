# MEKF Parameter Comparison

## Results: Parameters are IDENTICAL ✓

### mekf/main.py (Line 78)
```python
kalman_filter = MEKF(true_orientation.orientation, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1)
```

### main.py (Lines 77-83)
```python
mekf = MEKF(initial_quaternion, 
            constants['estimate_covariance'],      # 1.0
            constants['gyro_cov'],                  # 0.1
            constants['gyro_bias_cov'],             # 0.1
            constants['accel_proc_cov'],            # 0.1
            constants['accel_bias_cov'],            # 0.1
            constants['accel_obs_cov'])             # 0.1
```

### Parameter Mapping
| Parameter | Position | mekf/main.py | main.py | Match |
|-----------|----------|--------------|---------|-------|
| estimate_covariance | 1st | 1.0 | 1.0 | ✓ |
| gyro_cov | 2nd | 0.1 | 0.1 | ✓ |
| gyro_bias_cov | 3rd | 0.1 | 0.1 | ✓ |
| accel_proc_cov | 4th | 0.1 | 0.1 | ✓ |
| accel_bias_cov | 5th | 0.1 | 0.1 | ✓ |
| accel_obs_cov | 6th | 0.1 | 0.1 | ✓ |

**Conclusion**: The MEKF initialization parameters are 100% identical between the reference implementation and your main.py.

---

## New Test Case Added: random_rotation_motion

To debug potential issues with circular motion, a new test case has been added that exactly matches what `mekf/main.py` does:

### Characteristics
- **No position motion**: Vehicle stays at origin (0, 0, 0)
- **No acceleration**: Only angular velocity drives the attitude
- **Random angular velocity**: Applied every 100 time steps (scaled from mekf/main.py's 10 step interval)
- **Duration**: 20 seconds (2000 steps at dt=0.01s)
- **Attitude dynamics**: Pure rotation with no linear motion

### Purpose
This test case isolates attitude estimation from position estimation, allowing us to:
1. Verify MEKF works identically to `mekf/main.py` in your system
2. Identify if position divergence is due to MEKF issues or EKF/coordinate frame issues
3. Confirm the noise parameters are appropriate

### Test Flow
The test will be automatically included in the workflow:
1. `test_EKF_MEKF.py` generates `simulated_data_random_rotation_motion.csv`
2. `main.py` processes it and generates estimates
3. `compare_results.py` compares with ground truth

