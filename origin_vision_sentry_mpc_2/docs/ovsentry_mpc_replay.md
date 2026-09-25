# OVSentry MPC Replay

`ovsentry_mpc_replay` replays the recorded main-camera chain from
`ovsentry_omni_mpc`: YOLO, Tracker, Aimer, and world-frame sentry MPC planning.
It does not start ROS2, the industrial camera, any omniperception USB camera, or
send a gimbal command.

The recorder produces a pair with the same basename:

```text
records/2026-08-04_12-00-00.avi
records/2026-08-04_12-00-00.txt
```

When the text file is present, it provides the timestamp and IMU quaternion for
every recorded frame, which is required to replay tracking and world-frame MPC
correctly. An AVI without a matching text file also runs; it uses the video FPS
for timestamps and an identity orientation, so it is useful for checking the
main-camera detector path but not for judging world-frame tracking accuracy.

Build the target as usual, then run it from the repository root:

```bash
./build/ovsentry_mpc_replay records/2026-08-04_12-00-00 --config-path configs/sentry.yaml
```

The input can be either the shared basename or the `.avi` path. The matching
`.txt` file is found automatically. The config can be passed with
`--config-path` or as the second positional argument. Press `q` to stop playback.
Useful options:

```bash
# Process a selected interval with a chosen bullet speed.
./build/ovsentry_mpc_replay records/2026-08-04_12-00-00.avi \
  --config-path configs/sentry.yaml --start-index 300 --end-index 900 --bullet-speed 22

# Run without an OpenCV display window.
./build/ovsentry_mpc_replay records/2026-08-04_12-00-00 --no-display
```

The displayed MPC values are planner outputs. The replay only draws tracked
armor and aim-reference contours; it does not draw raw YOLO detections. The recording format does not
contain live small-yaw, big-yaw, pitch, or joint velocity feedback, so the replay
does not reproduce the controller takeover blend or emit firing/control commands.
