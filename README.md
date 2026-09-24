# Run instructions (BeetleBot, ROS 2 Jazzy)

## Sign → movement

| Sign | Movement |
|------|----------|
| Go Slow | creep forward: `linear.x=0.05`, `angular.z=0`, 3s |
| Speed Up | fast forward: `linear.x=0.25`, `angular.z=0`, 2s |
| Pedestrian Crossing | stop 3s, then forward `linear.x=0.08` for 2s |
| Road Closed | stop 1s, then reverse `linear.x=-0.1` for 2s |
| U-turn Ahead | 180° turn: `linear.x=0.1`, `angular.z=1.0`, π s |
| Roundabout Ahead | one full loop: `linear.x=0.12`, `angular.z=0.8`, 7.9s |

Movement is published on `/cmd_vel_nav` at 20 Hz, with a stop message sent when each action ends.

## 1. Clone on the robot

```
ssh veerobot@192.168.0.<bot_no>          # password: veerobot
git clone https://github.com/Arhaan-P/beetlebot_ai.git ~/beetlebot_ai
cd ~/beetlebot_ai
```
(Already cloned? `cd ~/beetlebot_ai && git pull` instead.)

## 2. Install dependencies (robot, one-time)

```
pip3 install --break-system-packages tensorflow numpy
```
Nothing else — no OpenCV, no extra ROS packages beyond what `robot.launch.py` already brings in (`rclpy`, `sensor_msgs`, `geometry_msgs`, `std_srvs`). `rqt_image_view` (used in step 5) ships with `ros-jazzy-desktop`; install it with `sudo apt install ros-jazzy-rqt-image-view` if it's missing.

## 3. Bring up the robot (terminal A)

```
source ~/lyra_ws/install/setup.bash
export ROS_DOMAIN_ID=<bot_no>
cd ~/lyra_ws
ros2 launch lyra_bringup robot.launch.py mode:=slam camera:=true
```

## 4. Run the detector (terminal B, on the robot)

```
source ~/lyra_ws/install/setup.bash
export ROS_DOMAIN_ID=<bot_no>
cd ~/beetlebot_ai
python3 sign_bot.py --dry-run   # first: check detection + window, robot does NOT move
python3 sign_bot.py             # then: for real, robot moves
```

If armed automatically fails, arm manually:
```
ros2 service call /lyra/arm std_srvs/srv/Trigger
```
Emergency stop:
```
ros2 service call /lyra/disarm std_srvs/srv/Trigger
```

## 5. View the camera window (PC)

```
source ~/lyra_ws/install/setup.bash
export ROS_DOMAIN_ID=<bot_no>
ros2 run rqt_image_view rqt_image_view /sign_bot/image
```
(If you SSH'd in with `ssh -X`, `sign_bot.py` opens this window itself — no need to run it separately.)

## What to expect

- The window shows the live camera feed with a coloured box + label (e.g. `Go Slow 97%`) around any detected sign, and a top bar showing the currently running action and its countdown.
- Hold a sign image steady in front of the camera for ~3 frames; the robot then runs that sign's movement (table above) and returns to looking for signs.
- The same sign is ignored for 4s after its action finishes, so it doesn't immediately re-trigger.
- `Ctrl+C` in terminal B stops the node and sends a stop command.

## Retrain the model (PC, not required to run the robot)

```
pip install tensorflow numpy pillow
python train.py
```
