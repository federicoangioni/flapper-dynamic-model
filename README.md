# flapper-dynamic-model

A dynamic model for the flapper+ flapping wing micro air vehicle, complete with controller, state estimation and power distribution.



## Data format


For the Nimble data, from the science paper "":

The data is divided in numbers of experiment, of which several runs are run. Below is the structure of the MATLAB structs.

experiment###
├── motion_tracking
│   ├── POSx, POSy, POSz
│   ├── POSx_aligned, POSy_aligned, POSz_aligned
│   ├── ROLL, PITCH, YAW, YAW_aligned
│   ├── SIDESLIP, COURSE, COURSE_aligned, TURN_RATE
│   ├── OMx, OMy, OMz
│   ├── ALPHx, ALPHy, ALPHz
│   ├── VEL_BODYx, VEL_BODYy, VEL_BODYz
│   ├── VEL_GROUNDx, VEL_GROUNDy, VEL_GROUNDz
│   ├── ACC_BODYx, ACC_BODYy, ACC_BODYz
│   ├── ACC_GROUNDx, ACC_GROUNDy, ACC_GROUNDz
│   ├── SPEED
│   ├── FILTERED signals
│   │   ├── DVEL_BODYx/y/z_filtered
│   │   ├── VEL_GROUNDx/y/z_filtered, VEL_GROUND*_aligned
│   │   ├── ACC_GROUNDx/y/z_filtered, ACC_GROUND*_aligned
│   │   ├── OMx/y/z_filtered
│   │   ├── ALPHx/y/z_filtered
│   │   ├── ACC_BODYx/y/z_filtered
│   │   ├── VEL_BODYx/y/z_filtered
│   ├── QUALITY
│   ├── DIHEDRAL
│   ├── ROLLfrom_rate_integration, PITCHfrom_rate_integration, YAWfrom_rate_integration
│   ├── TIME, fps, TIMEman
│   ├── Averages and stds (POS, VEL, ACC, ANGLES, etc.)
│
├── onboard
│   ├── angles_commands_setpoints
│   │   ├── TIME_onboard
│   │   ├── ROLL_IMU, PITCH_IMU, YAW_IMU, YAW_IMU_aligned
│   │   ├── CMDroll, CMDpitch, CMDyaw, CMDthrottle
│   │   ├── CMDleft_motor, CMDright_motor
│   │   ├── SETroll, SETpitch, SETyaw
│   │   ├── Filtered commands (CMD*_filtered)
│   │
│   ├── rates
│   │   ├── TIME_onboard_rates
│   │   ├── OMx/OMy/OMz_IMU (+ filtered)
│   │   ├── ALPHx/ALPHy/ALPHz_IMU (+ filtered)
│   │
│   ├── frequency
│       ├── TIME_onboard_freq
│       ├── FREQright_wing
│
├── onboard_interpolated
│   ├── CMDroll/pitch/yaw/throttle_interp
│   ├── CMDleft_motor_interp, CMDright_motor_interp
│   ├── SETroll/pitch/yaw_interp
│   ├── Filtered interpolated commands (CMD*_filtered_interp)
│   ├── FREQright_wing_interp
│   ├── TIME
│   ├── Averages and stds (for all *_interp signals)
