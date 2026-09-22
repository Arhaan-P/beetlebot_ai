import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2


class CameraTest(Node):

    def __init__(self):
        super().__init__("camera_test")

        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image,
            "/pi_camera/image_raw",
            self.image_callback,
            10
        )

        self.get_logger().info("BeetleBot camera test started")
        self.get_logger().info("Waiting for camera frames...")

    def image_callback(self, msg):

        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )

            cv2.imshow("BeetleBot Camera", frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(
                f"Camera conversion error: {e}"
            )


def main(args=None):

    rclpy.init(args=args)

    node = CameraTest()

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
