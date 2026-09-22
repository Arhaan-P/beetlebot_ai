import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import csv
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import tensorflow as tf
import numpy as np
import cv2

import math
from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "/home/student/beetlebot_ai/traffic_sign_model.h5"
CSV_PATH = "/home/student/beetlebot_ai/traffic_sign.csv"

CAMERA_TOPIC = "/pi_camera/image_raw"
CMD_VEL_TOPIC = "/cmd_vel"

COOLDOWN_SEC = 3.0


# ============================================================
# SIGN -> MOVEMENT MAPPING
#
# The trained model only knows the 59 classes in traffic_sign.csv,
# which do not include Go Slow / Speed Up / Pedestrian Crossing /
# Road Closed / U-turn Ahead / Roundabout Ahead. Each is mapped to
# the closest existing ClassId as a stand-in for the demo.
#
# ClassId: (action_name, linear_x, angular_z, duration_sec)
# duration_sec is None for actions that hold until the next trigger.
# ============================================================

UTURN_ANGULAR_SPEED = 1.0  # rad/s
UTURN_DURATION = math.pi / UTURN_ANGULAR_SPEED  # ~3.14s for 180 degrees

SIGN_ACTIONS = {
    18: ("go_slow", 0.10, 0.0, 3.0),              # stand-in: Maximum speed limit (90 km/h)
    19: ("speed_up", 0.30, 0.0, 3.0),             # stand-in: Maximum speed limit (110 km/h)
    7:  ("pedestrian_crossing", 0.0, 0.0, 3.0),   # stand-in: No entry for pedestrians
    1:  ("road_closed", 0.0, 0.0, None),          # stand-in: No entry
    23: ("uturn_ahead", 0.0, UTURN_ANGULAR_SPEED, UTURN_DURATION),  # stand-in: Turn left
    44: ("roundabout_ahead", 0.15, 0.4, 4.0),     # Roundabout
}


# ============================================================
# LOAD CSV
# ============================================================

csv_names = {}

with open(CSV_PATH, "r", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for row in reader:

        class_id = int(row["ClassId"])
        csv_names[class_id] = row["Name"]


# ============================================================
# IMPORTANT:
# RECREATE THE CLASS ORDER USED BY
# image_dataset_from_directory()
# ============================================================

# Your dataset has folders 0-58,
# but class 39 has no images.
#
# TensorFlow sorts folder names alphabetically.
# Therefore:
#
# 0, 1, 10, 11, 12, ..., 19, 2, 20, ...
#
# This must match the model's output neurons.

folder_names = [
    str(i)
    for i in range(59)
    if i != 39
]

folder_names = sorted(folder_names)


# Map MODEL OUTPUT INDEX → ORIGINAL CLASS ID

model_index_to_class_id = {
    index: int(folder_name)
    for index, folder_name in enumerate(folder_names)
}


# ============================================================
# LOAD MODEL
# ============================================================

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

IMG_SIZE = model.input_shape[1]


# ============================================================
# ROS NODE
# ============================================================

class TrafficSignDetector(Node):

    def __init__(self):

        super().__init__(
            "traffic_sign_detector",
            enable_rosout=False
        )

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            CAMERA_TOPIC,
            self.image_callback,
            10
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            CMD_VEL_TOPIC,
            10
        )

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


    def _trigger_action(self, class_id):

        name, linear_x, angular_z, duration = SIGN_ACTIONS[class_id]

        if self._action_timer is not None:
            self._action_timer.cancel()
            self._action_timer = None

        self._publish_twist(linear_x, angular_z)
        self.get_logger().info(f"Sign action: {name} (class {class_id})")

        if duration is not None:
            self._action_timer = self.create_timer(duration, self._end_action)


    def image_callback(self, msg):

        try:

            # ------------------------------------------------
            # ROS → OpenCV
            # ------------------------------------------------

            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )

            h, w = frame.shape[:2]


            # ------------------------------------------------
            # CENTER CROP
            # ------------------------------------------------

            crop_size = int(
                min(h, w) * 0.70
            )

            x1 = (w - crop_size) // 2
            y1 = (h - crop_size) // 2

            x2 = x1 + crop_size
            y2 = y1 + crop_size

            sign_crop = frame[
                y1:y2,
                x1:x2
            ]


            # ------------------------------------------------
            # BGR → RGB
            # ------------------------------------------------

            sign_crop = cv2.cvtColor(
                sign_crop,
                cv2.COLOR_BGR2RGB
            )


            # ------------------------------------------------
            # RESIZE
            # ------------------------------------------------

            img = cv2.resize(
                sign_crop,
                (IMG_SIZE, IMG_SIZE)
            )


            # ------------------------------------------------
            # IMPORTANT:
            # DO NOT NORMALIZE HERE.
            #
            # The model already contains:
            # Rescaling(1./255)
            # ------------------------------------------------

            img = img.astype(
                np.float32
            )

            img = np.expand_dims(
                img,
                axis=0
            )


            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            probabilities = model.predict(
                img,
                verbose=0
            )[0]


            # ------------------------------------------------
            # TOP PREDICTION
            # ------------------------------------------------

            model_index = int(
                np.argmax(probabilities)
            )

            confidence = float(
                probabilities[model_index]
            )


            # Convert model index → original ClassId

            class_id = model_index_to_class_id.get(
                model_index
            )


            # Get actual sign name from CSV

            sign_name = csv_names.get(
                class_id,
                f"Class {class_id}"
            )


            # ------------------------------------------------
            # TRIGGER MOVEMENT
            # ------------------------------------------------

            if confidence >= 0.50 and class_id in SIGN_ACTIONS:

                now = self.get_clock().now().nanoseconds / 1e9

                if (
                    class_id != self.last_action_class
                    or (now - self.last_action_time) > COOLDOWN_SEC
                ):
                    self._trigger_action(class_id)
                    self.last_action_class = class_id
                    self.last_action_time = now


            # ------------------------------------------------
            # DRAW ROI
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 0),
                2
            )


            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            if confidence >= 0.50:

                text = (
                    f"{sign_name} "
                    f"{confidence * 100:.1f}%"
                )

                cv2.putText(
                    frame,
                    text,
                    (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

            else:

                text = (
                    f"Low confidence "
                    f"{confidence * 100:.1f}%"
                )

                cv2.putText(
                    frame,
                    text,
                    (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA
                )


            cv2.imshow(
                "BeetleBot Traffic Sign AI",
                frame
            )

            cv2.waitKey(1)


        except Exception:
            pass


# ============================================================
# MAIN
# ============================================================

def main():

    rclpy.init()

    node = TrafficSignDetector()

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
