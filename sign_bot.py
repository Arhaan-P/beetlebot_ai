"""BeetleBot traffic-sign driver: camera -> CNN sign detection -> drive via /cmd_vel_nav.

No OpenCV: images are decoded with numpy, detection is two TensorFlow CNNs (detect.py), and the
annotated feed is published on /sign_bot/image and shown with rqt_image_view.

Run on the robot:  python3 ~/beetlebot_ai/sign_bot.py            (drives the robot)
                   python3 ~/beetlebot_ai/sign_bot.py --dry-run  (window + logs only, no movement)
"""
import math
import os
import shutil
import subprocess
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

from detect import LABELS, draw, find_sign, load_models, put_text

CAMERA_TOPICS = ["/pi_camera/image_raw", "/camera/image_raw"]  # whichever one the bringup publishes
CMD_VEL_TOPIC = "/cmd_vel_nav"
VIEW_TOPIC = "/sign_bot/image"   # annotated feed: boxes, labels, current action
RATE_HZ = 20          # same as `ros2 topic pub -r 20`
STABLE_FRAMES = 3     # sign must be seen this many frames in a row before acting
COOLDOWN_SEC = 4.0    # same sign is ignored this long after its action finishes
DRY_RUN = "--dry-run" in sys.argv

# sign -> steps of (linear.x m/s, angular.z rad/s, seconds). angular.z +ve = left.
# Calibration knobs: tune speeds/durations on the real robot.
ACTIONS = {
    "go_slow":             [(0.05, 0.0, 3.0)],                     # creep forward
    "speed_up":            [(0.25, 0.0, 2.0)],                     # fast forward (= -r 20 -t 40)
    "pedestrian_crossing": [(0.0, 0.0, 3.0), (0.08, 0.0, 2.0)],    # stop and wait, then proceed
    "road_closed":         [(0.0, 0.0, 1.0), (-0.1, 0.0, 2.0)],    # stop, then back away
    "u_turn_ahead":        [(0.1, 1.0, math.pi)],                  # 180 deg arc
    "roundabout_ahead":    [(0.12, 0.8, 2 * math.pi / 0.8)],       # one full loop
}


def to_rgb(msg):
    """sensor_msgs/Image -> HxWx3 uint8 RGB, numpy only."""
    h, w, enc = msg.height, msg.width, msg.encoding.lower()
    buf = np.frombuffer(msg.data, np.uint8).reshape(h, msg.step)
    if enc in ("rgb8", "bgr8", "rgba8", "bgra8"):
        img = buf[:, :w * len(enc[:-1])].reshape(h, w, -1)[..., :3]
        return img if enc.startswith("rgb") else img[..., ::-1]
    if enc == "mono8":
        return np.repeat(buf[:, :w, None], 3, axis=2)
    if enc in ("yuv422_yuy2", "yuyv", "yuv422", "uyvy"):
        p = buf[:, :w * 2].reshape(h, w // 2, 4).astype(np.float32)
        if enc in ("yuv422_yuy2", "yuyv"):
            y, u, v = p[..., [0, 2]], p[..., 1], p[..., 3]
        else:  # ROS "yuv422" is UYVY
            y, u, v = p[..., [1, 3]], p[..., 0], p[..., 2]
        y = y.reshape(h, w)
        u = np.repeat(u, 2, axis=1) - 128
        v = np.repeat(v, 2, axis=1) - 128
        rgb = np.stack([y + 1.402 * v, y - 0.344 * u - 0.714 * v, y + 1.772 * u], axis=-1)
        return np.clip(rgb, 0, 255).astype(np.uint8)
    raise ValueError(f"unsupported image encoding {msg.encoding}")


class SignBot(Node):

    def __init__(self):
        super().__init__("sign_bot")
        self.locator, self.classifier = load_models()
        self.pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)
        self.view_pub = self.create_publisher(Image, VIEW_TOPIC, 1)
        for topic in CAMERA_TOPICS:
            self.create_subscription(Image, topic, self.on_image, qos_profile_sensor_data)
        self.create_timer(1.0 / RATE_HZ, self.tick)

        self.action = None      # sign whose action is running
        self.steps = []         # remaining steps of that action
        self.step_end = 0.0
        self.streak_name, self.streak = None, 0
        self.finished_at = {}   # sign -> time its last action finished
        self.last_frame_time = time.monotonic()

        if DRY_RUN:
            self.get_logger().warn("DRY RUN: nothing will be published on " + CMD_VEL_TOPIC)
        else:
            self.arm()
        self.get_logger().info(f"Waiting for camera on {CAMERA_TOPICS}, driving via {CMD_VEL_TOPIC}")

    def arm(self):
        client = self.create_client(Trigger, "/lyra/arm")
        if not client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn("/lyra/arm not available; arm manually if the bot doesn't move")
            return
        client.call_async(Trigger.Request()).add_done_callback(
            lambda f: self.get_logger().info(f"Arm result: {f.result()}"))

    def send(self, linear_x, angular_z):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        if not DRY_RUN:
            self.pub.publish(msg)

    def start(self, name):
        self.action = name
        self.steps = list(ACTIONS[name])
        self.step_end = time.monotonic() + self.steps[0][2]
        self.get_logger().info(f"Sign: {LABELS[name]} -> {ACTIONS[name]}")

    def tick(self):
        """20 Hz: publish the current step of the running action."""
        if self.action is None:
            return
        now = time.monotonic()
        if now >= self.step_end:
            self.steps.pop(0)
            if not self.steps:
                self.send(0.0, 0.0)
                self.get_logger().info(f"Done: {LABELS[self.action]}")
                self.finished_at[self.action] = now
                self.action = None
                return
            self.step_end = now + self.steps[0][2]
        linear_x, angular_z, _ = self.steps[0]
        self.send(linear_x, angular_z)

    def on_image(self, msg):
        try:
            frame = to_rgb(msg)
            det = find_sign(frame, self.locator, self.classifier)

            top = det[1] if det else None
            self.streak = self.streak + 1 if top is not None and top == self.streak_name else int(top is not None)
            self.streak_name = top
            now = time.monotonic()
            if (top is not None and self.streak >= STABLE_FRAMES and self.action is None
                    and now - self.finished_at.get(top, -1e9) > COOLDOWN_SEC):
                self.start(top)

            self.publish_view(frame, det, now, msg.header)
        except Exception as e:  # keep the node alive on a bad frame
            self.get_logger().error(f"Frame processing error: {e}")

    def publish_view(self, frame, det, now, header):
        img = frame.copy()
        if img.shape[1] < 640:  # small camera frames: upscale so labels stay readable
            k = -(-640 // img.shape[1])
            img = img.repeat(k, axis=0).repeat(k, axis=1)
            if det:
                det = (tuple(v * k for v in det[0]), det[1], det[2])
        banner = 30
        draw(img, det, top_margin=banner)

        if self.action is not None:
            left = sum(s[2] for s in self.steps[1:]) + max(0.0, self.step_end - now)
            status, color = f" ACTION: {LABELS[self.action]} ({left:.1f}s)", (0, 255, 0)
        else:
            status, color = " Looking for signs...", (220, 220, 220)
        if DRY_RUN:
            status += "  [DRY RUN]"
        fps = 1.0 / max(1e-3, now - self.last_frame_time)
        self.last_frame_time = now
        img[:banner] = 0
        put_text(img, status, 0, 5, color)
        put_text(img, f"{fps:4.1f} fps ", img.shape[1] - 110, 5, (200, 200, 200))

        out = Image()
        out.header = header
        out.height, out.width = img.shape[:2]
        out.encoding = "rgb8"
        out.step = img.shape[1] * 3
        out.data = np.ascontiguousarray(img).tobytes()
        self.view_pub.publish(out)


def open_window(node):
    """Open the camera window (rqt_image_view on the annotated feed) if there's a display."""
    if os.environ.get("DISPLAY") and shutil.which("ros2"):
        node.get_logger().info(f"Opening window: rqt_image_view {VIEW_TOPIC}")
        return subprocess.Popen(["ros2", "run", "rqt_image_view", "rqt_image_view", VIEW_TOPIC])
    node.get_logger().warn(f"No display here. On your PC run: ros2 run rqt_image_view rqt_image_view {VIEW_TOPIC}")
    return None


def main():
    rclpy.init()
    node = SignBot()
    window = open_window(node)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.send(0.0, 0.0)  # never leave the bot driving
        except Exception:
            pass
        if window:
            window.terminate()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
