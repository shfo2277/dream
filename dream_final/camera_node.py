
# 실행코드 ros2 run camera_ros camera
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.publisher_ = self.create_publisher(Image, 'camera/image_raw', 30)
        self.bridge = CvBridge()
        self.timer = self.create_timer(0.1, self.publish_image)  # 10Hz
        self.cap = cv2.VideoCapture(0)  # 0번 카메라 사용

        if not self.cap.isOpened():
            self.get_logger().error("카메라를 열 수 없습니다.")
            rclpy.shutdown()

    def publish_image(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("카메라 프레임을 읽을 수 없습니다.")
            return
        
        # OpenCV 이미지를 ROS 이미지 메시지로 변환
        image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.publisher_.publish(image_msg)
        self.get_logger().info("이미지 퍼블리시 완료")

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    camera_node = CameraNode()
    
    try:
        rclpy.spin(camera_node)
    except KeyboardInterrupt:
        pass
    finally:
        camera_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
