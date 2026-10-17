#이거는 카메라의 상위 35프로에 떠있는 라이다의 잘못 투영된 점들을 무시함
#시각적으로 욜로와 라이다 통합을 확인할 수 있는 코드
#근데 잘 안됨요~ 라이다 거리랑 각도 잘 안 맞음
import rclpy
from rclpy.node import Node
import numpy as np
import json
from sensor_msgs.msg import LaserScan, Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2

class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')
        self.bridge = CvBridge()

        # Subscribers for YOLO bounding boxes, LiDAR data, and camera images
        self.yolo_sub = self.create_subscription(String, '/yolov5/detections', self.yolo_callback, 10)
        self.lidar_sub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.camera_sub = self.create_subscription(Image, '/camera/image_raw', self.camera_callback, 10)

        # Publisher for fusion results
        self.result_pub = self.create_publisher(String, '/fusion_results', 10)

        # Calibration parameters
        self.camera_intrinsics = np.array([
            [390.20006229, 0., 321.54814476, 0.],
            [0., 390.02222138, 238.40380784, 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]
        ])

        self.extrinsics = np.array([
            [0.01138116, -0.99993511, 0.00050179, -0.05379494],
            [-0.21328417, -0.00291786, -0.97698585, 0.44551191],
            [0.97692391, 0.01101221, -0.21330353, -0.36708386],
            [0.00000000, 0.00000000, 0.00000000, 1.00000000]
        ])

        # Fusion results
        self.lidar_points = None
        self.yolo_detections = []
        self.current_frame = None  # Latest camera frame

    def yolo_callback(self, msg):
        """Callback to receive YOLO bounding box detections."""
        try:
            self.yolo_detections = json.loads(msg.data)['detections']
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse YOLO detections: {e}")

        # Process detections if LiDAR data is available
        if self.lidar_points is not None:
            self.process_fusion()

    def lidar_callback(self, msg):
        """Callback to receive LiDAR scan data."""
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        ranges = msg.ranges
        self.lidar_points = []

        for i, r in enumerate(ranges):
            angle = angle_min + i * angle_increment
            if msg.range_min < r < msg.range_max:
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                self.lidar_points.append([x, y, 0])  # Assume height (z) = 0

        self.lidar_points = np.array(self.lidar_points)

    def camera_callback(self, msg):
        """Callback to receive camera images."""
        self.current_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def process_fusion(self):
        """Process YOLO and LiDAR data to calculate obstacle distance and angle."""
        if self.current_frame is None:
            return  # Wait for a camera frame

        frame = self.current_frame.copy()

        for detection in self.yolo_detections:
            bbox = detection['bbox']  # Bounding box (x1, y1, x2, y2)
            object_name = detection['class_name'] 

            # Transform LiDAR points to camera coordinates
            lidar_points_camera = self.transform_lidar_to_camera()

            # Filter out points in the top 35% depth
            lidar_points_camera = self.filter_points_by_depth(lidar_points_camera)

            # Extract LiDAR points within the bounding box
            lidar_points_in_bbox = self.get_lidar_points_in_bbox(bbox, lidar_points_camera)

            if len(lidar_points_in_bbox) > 0:
                # Calculate average distance and angle
                avg_distance = np.mean(np.linalg.norm(lidar_points_in_bbox[:, :2], axis=1))
                avg_angle = np.mean(np.arctan2(lidar_points_in_bbox[:, 1], lidar_points_in_bbox[:, 0]))

                # Determine direction based on angle thresholds
                if -0.1 < avg_angle < 0.1:  # Close to 0 radians
                    direction = 'center'
                elif avg_angle > 0:  # Positive angle
                    direction = 'left'
                else:  # Negative angle
                    direction = 'right'

                # Publish the result
                result_msg = String()
                result_msg.data = f"Object: {object_name}, Distance: {avg_distance:.2f}m, Direction: {direction}"
                self.result_pub.publish(result_msg)

            # Visualize the bounding box and projected points
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            for point in lidar_points_in_bbox:
                px, py, pz, _ = self.camera_intrinsics @ point  # Project point
                px, py = int(px / pz), int(py / pz)  # Normalize
                cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)

        # Display the frame
        cv2.imshow("Fusion Visualization", frame)
        cv2.waitKey(1)

    def transform_lidar_to_camera(self):
        """Transform LiDAR points to the camera coordinate system."""
        points_homogeneous = np.hstack((self.lidar_points, np.ones((self.lidar_points.shape[0], 1))))  # Make homogeneous
        transformed_points = self.extrinsics @ points_homogeneous.T  # Apply extrinsics
        return transformed_points.T  # Return transformed points in [x, y, z, 1]

    def filter_points_by_depth(self, points):
        """Filter out points in the top 35% of depth."""
        depths = np.linalg.norm(points[:, :3], axis=1)  # Calculate depth (3D norm)
        threshold = np.percentile(depths, 65)  # 65th percentile (keep 65% closest)
        return points[depths <= threshold]

    def get_lidar_points_in_bbox(self, bbox, lidar_points_camera):
        """Extract LiDAR points that fall within the YOLO bounding box."""
        x1, y1, x2, y2 = bbox
        lidar_points_in_bbox = []

        for point in lidar_points_camera:
            x, y, z, _ = self.camera_intrinsics @ point  # Project to image plane
            x /= z  # Normalize by depth
            y /= z

            if x1 <= x <= x2 and y1 <= y <= y2:  # Check if projected point falls in bounding box
                lidar_points_in_bbox.append(point)

        return np.array(lidar_points_in_bbox)


def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

