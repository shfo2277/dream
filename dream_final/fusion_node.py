"""
fusion_node – LiDAR + YOLO 센서 퓨전 (장애물 거리·방향)

LiDAR 스캔 점을 외부 파라미터로 카메라 좌표로 바꾸고 내부 파라미터로 영상에 투영한다.
YOLO 바운딩박스 안에 들어온 점(먼 쪽 35% 제외)을 다시 LiDAR 좌표로 돌려 평균 거리와 각도를 구하고,
각도로 방향을 정한다. (±15° center, 15~40° left/right, 그 밖은 out of range)
  sub: /yolov5/detections, /scan (sensor_msgs/LaserScan), /camera/image_raw
  pub: /fusion_results (std_msgs/String) "Object: bollard, Distance: 1.23m, Direction: center"
"""
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
                # Convert back to LiDAR coordinates to calculate distance
                lidar_points_original = self.transform_camera_to_lidar(lidar_points_in_bbox)

                # Calculate average distance and angle in original LiDAR coordinates
                avg_distance = np.mean(np.linalg.norm(lidar_points_original[:, :2], axis=1))
                avg_angle = np.mean(np.arctan2(lidar_points_original[:, 1], lidar_points_original[:, 0]))

                # Convert angle to degrees for easier thresholding
                avg_angle_deg = np.degrees(avg_angle)

                # Determine direction based on angle thresholds
                if -15 <= avg_angle_deg <= 15:  # Central
                    direction = 'center'
                elif -40 <= avg_angle_deg < -15:  # Right
                    direction = 'right'
                elif 15 < avg_angle_deg <= 40:  # Left
                    direction = 'left'
                else:
                    direction = 'out of range'

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
        points_homogeneous = np.hstack((self.lidar_points, np.ones((self.lidar_points.shape[0], 1))))  # Homogeneous
        transformed_points = self.extrinsics @ points_homogeneous.T  # Apply extrinsics
        return transformed_points.T  # Return transformed points in [x, y, z, 1]

    def transform_camera_to_lidar(self, points_camera):
        """
        Transform camera points back to LiDAR coordinates.
        :param points_camera: Points in the camera coordinate system.
        :return: Points transformed back to the LiDAR coordinate system.
        """
        inv_extrinsics = np.linalg.inv(self.extrinsics)  # Inverse of extrinsics
        points_homogeneous = np.hstack((points_camera[:, :3], np.ones((points_camera.shape[0], 1))))  # Homogeneous
        transformed_points = inv_extrinsics @ points_homogeneous.T  # Apply inverse transform
        return transformed_points.T  # Return points in LiDAR coordinates

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

