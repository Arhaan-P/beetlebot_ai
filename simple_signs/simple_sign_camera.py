import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger

import tensorflow as tf
import numpy as np
import cv2


MODEL_PATH = "/home/veerobot/beetlebot_ai/simple_signs/simple_sign_model.h5"

CAMERA_TOPIC = "/pi_camera/image_raw"
CMD_VEL_TOPIC = "/cmd_vel"

IMG_SIZE = 64
CONFIDENCE_THRESHOLD = 0.60
COOLDOWN_SEC = 3.0

CLASSES = ["background", "forward", "left", "stop"]

# class_name: (linear_x, angular_z, duration_sec)
# duration_sec is None for actions that hold until the next trigger.
SIGN_ACTIONS = {
    "forward": (0.15, 0.0, 3.0),
    "left": (0.05, 1.0, 3.0),
    "stop": (0.0, 0.0, None),
}

model = tf.keras.models.load_model(MODEL_PATH, compile=False)


class SimpleSignDetector(Node):

    def __init__(self):
        super().__init__("simple_sign_detector", enable_rosout=False)

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image, CAMERA_TOPIC, self.image_callback, 10
        )
        self.cmd_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10)

        self.last_action_class = None
        self.last_action_time = 0.0
        self._action_timer = None

        self._arm_robot()

    def _arm_robot(self):
        client = self.create_client(Trigger, "/lyra/arm")

        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn("Arm service /lyra/arm not available")
            return

        future = client.call_async(Trigger.Request())
        future.add_done_callback(
            lambda f: self.get_logger().info(f"Arm result: {f.result()}")
        )

    def _publish_twist(self, linear_x, angular_z):
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.cmd_pub.publish(msg)

    def _end_action(self):
        self._publish_twist(0.0, 0.0)
        if self._action_timer is not None:
            self._action_timer.cancel()
            self._action_timer = None

    def _trigger_action(self, class_name):
        linear_x, angular_z, duration = SIGN_ACTIONS[class_name]

        if self._action_timer is not None:
            self._action_timer.cancel()
            self._action_timer = None

        self._publish_twist(linear_x, angular_z)
        self.get_logger().info(f"Sign action: {class_name}")

        if duration is not None:
            self._action_timer = self.create_timer(duration, self._end_action)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            h, w = frame.shape[:2]

            crop_size = int(min(h, w) * 0.70)
            x1 = (w - crop_size) // 2
            y1 = (h - crop_size) // 2
            x2 = x1 + crop_size
            y2 = y1 + crop_size
            crop = frame[y1:y2, x1:x2]

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            img = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
            img = np.expand_dims(img, axis=0)

            logits = model.predict(img, verbose=0)[0]
            probs = tf.nn.softmax(logits).numpy()

            class_index = int(np.argmax(probs))
            confidence = float(probs[class_index])
            class_name = CLASSES[class_index]

            if confidence >= CONFIDENCE_THRESHOLD and class_name in SIGN_ACTIONS:
                now = self.get_clock().now().nanoseconds / 1e9

                if (
                    class_name != self.last_action_class
                    or (now - self.last_action_time) > COOLDOWN_SEC
                ):
                    self._trigger_action(class_name)
                    self.last_action_class = class_name
                    self.last_action_time = now

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 0, 255)
            text = f"{class_name} {confidence * 100:.1f}%"
            cv2.putText(
                frame, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 2, cv2.LINE_AA
            )

            cv2.imshow("BeetleBot Simple Sign AI", frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Frame processing error: {e}")


def main():
    rclpy.init()
    node = SimpleSignDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
