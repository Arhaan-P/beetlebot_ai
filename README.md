# Running this on BeetleBot (ROS2 Jazzy lab)

Two terminals are used: your **ros2-jazzy** terminal (stays on your PC) and the
**robot terminal** (what you land in after `ssh`). Each step below says which one.

1. **[ros2-jazzy]** Connect to Wi-Fi **BEETLEBOT_5G** (password `15619xxx`).

2. **[ros2-jazzy]** SSH into the robot (this opens the robot terminal):
   ```
   ssh veerobot@192.168.0.<bot_no>
   ```
   (password `veerobot`)

3. **[robot terminal]** Source the workspace and set the domain ID:
   ```
   source ~/lyra_ws/install/setup.bash
   export ROS_DOMAIN_ID=<bot_no>
   ```

4. **[ros2-jazzy]** In a separate, non-SSH terminal on your PC, run the same two
   commands (same `<bot_no>`) so your PC can see the robot.

5. **[robot terminal]** Bring up the robot:
   ```
   cd ~/lyra_ws
   ros2 launch lyra_bringup robot.launch.py mode:=slam lidar:=true camera:=true
   ```

6. **[ros2-jazzy]** Copy the code to the robot:
   ```
   scp traffic_sign_camera.py traffic_sign.csv traffic_sign_model.h5 veerobot@192.168.0.<bot_no>:/home/student/beetlebot_ai/
   ```

7. **[ros2-jazzy]** Verify topics are live:
   ```
   ros2 topic list
   ```
   Confirm `/pi_camera/image_raw` and `/cmd_vel` are present.

8. **[robot terminal]** Run the detector:
   ```
   python3 /home/student/beetlebot_ai/traffic_sign_camera.py
   ```
   It arms the robot automatically. If arming fails, arm manually:
   ```
   ros2 service call /lyra/arm std_srvs/srv/Trigger
   ```

9. Hold a traffic sign image on your phone in front of the BeetleBot's camera and watch it detect + move.
