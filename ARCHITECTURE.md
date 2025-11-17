# System Architecture: Dual Filter Estimation

## Block Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA ACQUISITION                               │
│                        data.csv (IMU + GPS)                             │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼                           ▼
        ┌──────────────────────┐    ┌──────────────────────┐
        │  Vector Extraction   │    │  Vector Extraction   │
        │  ─────────────────   │    │  ─────────────────   │
        │  accel_meas = [ax,   │    │  gyro_meas = [gx,    │
        │             ay, az]  │    │            gy, gz]   │
        └─────────┬────────────┘    └────────┬─────────────┘
                  │                          │
        ┌─────────▼──────────────────────────▼─────────┐
        │         MAIN PROCESSING LOOP                  │
        │     for i in range(len(time)):                │
        └─────────┬──────────────────────────┬─────────┘
                  │                          │
    ┌─────────────▼─────────────┐  ┌────────▼──────────────┐
    │     EKF POSITION FILTER   │  │   MEKF ATTITUDE FILTER│
    │  ═══════════════════════  │  │  ═════════════════════│
    │                           │  │                       │
    │ Input:                    │  │ Input:                │
    │  - Position (from GPS)    │  │  - Gyroscope data     │
    │  - Acceleration (accel_m) │  │  - Accelerometer data │
    │                           │  │                       │
    │ Prediction:               │  │ Prediction:           │
    │  x_k+1 = A·x_k + B·u_k   │  │  - Quaternion         │
    │  P_k+1 = A·P_k·A' + Q    │  │    integration         │
    │                           │  │  - Error propagation  │
    │ Update:                   │  │                       │
    │  - Kalman gain calc       │  │ Update:               │
    │  - State correction       │  │  - Kalman gain        │
    │  - Covariance update      │  │  - Quaternion update  │
    │                           │  │  - Bias correction    │
    │ Output:                   │  │                       │
    │  x_k = [9×1]              │  │ Output:               │
    │  P_k = [9×9]              │  │  q = Quaternion       │
    │  (position estimates)     │  │  P_error = [15×15]    │
    │                           │  │  (attitude quaternion)│
    └─────────┬─────────────────┘  └────────┬──────────────┘
              │                             │
              │    ┌──────────────────────┐ │
              └────┤  Storage Update      ├─┘
                   │  ──────────────────  │
                   │  x_k_storage[:,i]    │
                   │  quaternion_storage  │
                   │     append(q)        │
                   └──────────────────────┘
                             │
                             │ (After all iterations)
                             ▼
                   ┌──────────────────────┐
                   │  RESULTS SAVING      │
                   │  ──────────────────  │
                   │  - EKF estimates     │
                   │  - EKF covariance    │
                   │  - MEKF quaternions  │
                   │  - MEKF covariance   │
                   └──────────┬───────────┘
                              ▼
                   ┌──────────────────────┐
                   │  OUTPUT FILES        │
                   │  ──────────────────  │
                   │  results/            │
                   │  ├─ ekf_*.npy        │
                   │  ├─ mekf_*.npy       │
                   │  └─ mekf_*.txt       │
                   └──────────────────────┘
```

## Detailed Signal Flow per Iteration

```
Time Step i:
─────────────────────────────────────────────────────────────────

1. DATA EXTRACTION
   ─────────────────
   From arrays at index [i]:
   
   accel_meas ← [ax[i], ay[i], az[i]]
   gyro_meas  ← [gx[i], gy[i], gz[i]]
   
   GPS position ← [x[i], y[i], z[i]]


2. PARALLEL PROCESSING
   ───────────────────────────────────────
   
   ┌─ EKF Branch ──┐          ┌─ MEKF Branch ─┐
   │               │          │               │
   │ 1. Prediction │          │ 1. Prediction │
   │    (using A)  │          │    (using gyro│
   │               │          │     + A matrix)│
   │ 2. Update     │          │               │
   │    (using GPS │          │ 2. Update     │
   │     + accel)  │          │    (using     │
   │               │          │     accel)    │
   └─ x_k, P_k ───┘          └─ q, P_error ──┘
   

3. STORAGE UPDATE
   ─────────────────
   x_k_storage[:, i]           ← x_k
   P_k_storage[:, :, i]        ← P_k
   quaternion_storage[i]       ← q
   mekf_cov_storage[:, :, i]   ← P_error


4. LOOP TO NEXT ITERATION (i+1)
   ─────────────────────────────
   x_k is now state at k+1
   q is now estimate at k+1
   (fed back for next iteration)
```

## State Vectors

### EKF State (9 elements)
```
x_k = [
    x,      # Position X (ECEF)
    y,      # Position Y (ECEF)
    z,      # Position Z (ECEF)
    vx,     # Velocity X
    vy,     # Velocity Y
    vz,     # Velocity Z
    ax,     # Acceleration X
    ay,     # Acceleration Y
    az      # Acceleration Z
]
```

### MEKF State (15 elements - Error State)
```
x_error = [
    δθ₁, δθ₂, δθ₃,         # Orientation error (0:3)
    δv₁, δv₂, δv₃,         # Velocity error (3:6)
    δp₁, δp₂, δp₃,         # Position error (6:9)
    ω_b₁, ω_b₂, ω_b₃,      # Gyro bias (9:12)
    a_b₁, a_b₂, a_b₃       # Accel bias (12:15)
]

Full state: q = Quaternion(δθ) + errors
```

## Matrix Dimensions Summary

### EKF Matrices
| Matrix | Size   | Purpose                      |
|--------|--------|------------------------------|
| A      | 9×9    | State transition             |
| B      | 9×3    | Control input (all zeros)    |
| C      | 6×9    | Measurement model            |
| Q      | 9×9    | Process noise covariance     |
| R      | 6×6    | Measurement noise covariance |
| K      | 9×6    | Kalman gain                  |

### MEKF Matrices
| Matrix | Size   | Purpose                      |
|--------|--------|------------------------------|
| F      | 15×15  | Error dynamics               |
| Φ      | 15×15  | State transition (discrete)  |
| H      | 3×15   | Measurement model            |
| Q      | 15×15  | Process noise covariance     |
| R      | 3×3    | Measurement noise covariance |
| K      | 15×3   | Kalman gain                  |

## Independence Guarantee

```
┌───────────────────────────────────────────────────────────┐
│           NO CROSS-COUPLING BETWEEN FILTERS               │
├───────────────────────────────────────────────────────────┤
│                                                            │
│  EKF State    ┌──┐    No shared variables    ┌──┐         │
│   (x_k)    ──┤  ├─────────────────────────────┤  ├──      │
│              └──┘                            └──┘         │
│                       ↓ INDEPENDENT ↓                     │
│  MEKF State   ┌──┐                         ┌──┐          │
│   (q, P_err)─┤  ├──────────────────────────┤  ├──       │
│              └──┘                            └──┘         │
│                                                            │
│  • Each filter maintains own state                        │
│  • Each filter maintains own covariance                   │
│  • No shared intermediate variables                       │
│  • Results could be fused downstream (optional)          │
│                                                            │
└───────────────────────────────────────────────────────────┘
```

## Processing Timeline

```
Loop Iteration Timeline (single time step dt):
──────────────────────────────────────────────

 t0: Read data[i]
     │
     ├─→ EKF Prediction    (1 ms)
     │   EKF Update        (1 ms)
     │   Store results     (0.1 ms)
     │
     └─→ MEKF Prediction   (1 ms)
         MEKF Update       (1 ms)
         Store results     (0.1 ms)

 t1: Move to i+1
     Repeat
     
Total per iteration: ~4 ms (parallelizable to ~2 ms)
```

## Output Storage Structure

```
results/
├── ekf_position_estimates.npy
│   └─ Shape: (9, N_timesteps)
│      Rows: [x, y, z, vx, vy, vz, ax, ay, az]
│
├── ekf_covariance.npy
│   └─ Shape: (9, 9, N_timesteps)
│      Each [:, :, i] is a covariance matrix
│
├── mekf_quaternions.txt
│   └─ CSV format
│      Columns: Time_Step, Qw, Qx, Qy, Qz
│
└── mekf_covariance.npy
    └─ Shape: (15, 15, N_timesteps)
       Error state covariances
```

## Filter Characteristics

### EKF (Position)
- **Type**: Extended Kalman Filter
- **State Space**: Continuous position/velocity/acceleration
- **Measurement Input**: GPS position + IMU acceleration
- **Output**: 9-element state vector
- **Time Constant**: dt = 0.01s
- **Tuning**: σ_GPS, σ_IMU, initial uncertainty

### MEKF (Attitude)
- **Type**: Multiplicative Extended Kalman Filter
- **State Space**: Quaternion + error covariance
- **Measurement Input**: Gyroscope + accelerometer
- **Output**: Quaternion (w, x, y, z)
- **Features**: Bias estimation & correction
- **Tuning**: Multiple noise covariance parameters

---

**Note**: For fusion of estimates, this architecture provides the foundation. Downstream processing can combine position from EKF with attitude from MEKF to create a complete 6DOF state estimate.

