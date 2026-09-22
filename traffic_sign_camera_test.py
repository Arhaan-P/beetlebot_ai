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


MODEL_PATH = "/home/veerobot/beetlebot_ai/traffic_sign_model.h5"
CSV_PATH = "/home/veerobot/beetlebot_ai/traffic_sign.csv"

CAMERA_TOPIC = "/pi_camera/image_raw"

IMG_SIZE = 32


# ------------------------------------------------------------
# Load class names
# ------------------------------------------------------------

class_names = {}

with open(CSV_PATH, "r", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for row in reader:
        class_names[int(row["ClassId"])] = row["Name"]


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


print("Model loaded")
print("Input :", model.input_shape)
print("Output:", model.output_shape)


# ------------------------------------------------------------
# ROS node
# ------------------------------------------------------------

class TrafficSignTest(Node):

    def __init__(self):

        super().__init__(
            "traffic_sign_test",
            enable_rosout=False
        )

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            CAMERA_TOPIC,
            self.callback,
            10
        )


    def callback(self, msg):

        try:

            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )

            h, w = frame.shape[:2]

            # ------------------------------------------------
            # Center crop
            # ------------------------------------------------

            crop_size = int(min(h, w) * 0.70)

            x1 = (w - crop_size) // 2
            y1 = (h - crop_size) // 2

            x2 = x1 + crop_size
            y2 = y1 + crop_size

            crop = frame[y1:y2, x1:x2]


            # ------------------------------------------------
            # BGR → RGB
            # ------------------------------------------------

            crop = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2RGB
            )


            # ------------------------------------------------
            # Resize
            # ------------------------------------------------

            img = cv2.resize(
                crop,
                (IMG_SIZE, IMG_SIZE)
            )


            img = img.astype(np.float32)

            # Normalize
            img = img / 255.0

            img = np.expand_dims(
                img,
                axis=0
            )


            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            probs = model.predict(
                img,
                verbose=0
            )[0]


            # Get TOP 5
            top5 = np.argsort(probs)[-5:][::-1]


            # ------------------------------------------------
            # Draw ROI
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 0),
                2
            )


            # ------------------------------------------------
            # Display TOP 5
            # ------------------------------------------------

            y = 35

            for rank, class_id in enumerate(top5):

                confidence = probs[class_id]

                name = class_names.get(
                    int(class_id),
                    f"Class {class_id}"
                )

                text = (
                    f"{rank + 1}. "
                    f"{name}: "
                    f"{confidence * 100:.1f}%"
                )

                cv2.putText(
                    frame,
                    text,
                    (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

                y += 25


            # ------------------------------------------------
            # Show
            # ------------------------------------------------

            cv2.imshow(
                "Traffic Sign Model Test",
                frame
            )

            cv2.waitKey(1)


        except Exception as e:
            self.get_logger().error(f"Frame processing error: {e}")


def main():

    rclpy.init()

    node = TrafficSignTest()

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
