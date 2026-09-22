# Running this on BeetleBot (ROS2 Jazzy lab)

## Machines

- **main PC** — your laptop/desktop, `student` user, home dir `/home/student`. Not part of this repo.
- **robot** — reached via `ssh veerobot@192.168.0.<bot_no>`, home dir `/home/veerobot`. This repo lives there at `/home/veerobot/beetlebot_ai`.

Two terminals are used: your **[main PC]** terminal (stays on your PC) and the
**[robot]** terminal (what you land in after `ssh`). Each step below says which one.

The detector script opens an OpenCV window (`cv2.imshow`) showing the live
camera feed with the detection ROI box, the predicted sign name, and
confidence — this requires a real display, so SSH needs X11 forwarding
(step 2 below) or the robot needs a monitor plugged in directly.

1. **[main PC]** Connect to Wi-Fi **BEETLEBOT_5G** (password `15619xxx`).

2. **[main PC]** SSH into the robot **with X11 forwarding** (this opens the robot terminal):
   ```
   ssh -X veerobot@192.168.0.<bot_no>
   ```
   (password `veerobot`). On Windows you also need an X server running on
   your PC first (e.g. [VcXsrv](https://sourceforge.net/projects/vcxsrv/),
   launched with "Disable access control" checked) — without one, `-X`
   connects fine but no window will ever appear.

3. **[robot]** Source the workspace and set the domain ID:
   ```
   source ~/lyra_ws/install/setup.bash
   export ROS_DOMAIN_ID=<bot_no>
   ```

4. **[main PC]** In a separate, non-SSH terminal on your PC, run the same two
   commands (same `<bot_no>`) so your PC can see the robot.

5. **[robot]** Bring up the robot:
   ```
   cd ~/lyra_ws
   ros2 launch lyra_bringup robot.launch.py mode:=slam lidar:=true camera:=true
   ```

6. **[robot]** Pull the latest code (repo already cloned at `~/beetlebot_ai`):
   ```
   cd ~/beetlebot_ai
   git pull
   ```
   First-time setup only, if the repo isn't there yet:
   ```
   git clone https://github.com/Arhaan-P/beetlebot_ai.git ~/beetlebot_ai
   ```

7. **[main PC]** Verify topics are live:
   ```
   ros2 topic list
   ```
   Confirm `/pi_camera/image_raw` and `/cmd_vel` are present.

8. **[robot]** Run the detector (in the same `-X` SSH terminal from step 2):
   ```
   python3 ~/beetlebot_ai/traffic_sign_camera.py
   ```
   A window titled "BeetleBot Traffic Sign AI" should pop up on your PC
   showing the live feed, the yellow detection box, and the predicted sign
   name/confidence. Log lines (arm result, triggered actions, errors) print
   in this same terminal.

   It arms the robot automatically. If arming fails, arm manually:
   ```
   ros2 service call /lyra/arm std_srvs/srv/Trigger
   ```

9. Hold a traffic sign image on your phone in front of the BeetleBot's camera and watch it detect + move.

## Troubleshooting: no camera window appears

- Confirm you SSH'd with `ssh -X` (not plain `ssh`) and an X server is
  running on your PC before you connected.
- Run `echo $DISPLAY` in the robot terminal — if it's empty, X11 forwarding
  isn't active; reconnect with `-X`.
- Check the terminal for `Frame processing error:` log lines — the script
  now logs these instead of failing silently.
