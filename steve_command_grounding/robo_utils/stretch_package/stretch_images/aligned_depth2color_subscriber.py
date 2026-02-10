import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.wait_for_message import wait_for_message
import cv2
from cv_bridge import CvBridge

class AlignedDepth2ColorSubscriber(Node):
    def __init__(self, topic, rotate=True, save=False, vis=False):
        super().__init__('aligned_depth2color_subscriber_node')
        self.bridge = CvBridge()
        self.rotate = rotate 
        self.save = save
        self.vis = vis
        self.image = None
        self.cv_image = None

        try:
            _, msg = wait_for_message(Image, self, topic, time_to_wait=10.0)
            if msg is not None:
                self.process_image(msg)
        except Exception as e:
            self.get_logger().error(f'Error waiting for image on {topic}: {e}')

    def process_image(self, msg):
        try:
            self.image = msg
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, '16UC1')
            if self.rotate:
                self.cv_image = cv2.rotate(self.cv_image, cv2.ROTATE_90_CLOCKWISE)
            if self.save:
                # Path might need workspace adaptation
                cv2.imwrite('/tmp/depth_image.png', self.cv_image)
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')
