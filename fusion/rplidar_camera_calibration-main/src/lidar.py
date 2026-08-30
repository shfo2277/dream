#!/usr/bin/env python3
# -- coding: utf-8 --
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from laser_geometry.laser_geometry import LaserProjection
import sensor_msgs.point_cloud2 as pc2

class RPLidarNode(Node):
    def __init__(self):
        super().__init__('rplidar_node')  # 노드 이름 설정
        self.laser_projector = LaserProjection()
        # '/scan' 토픽 구독
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        # '/pointcloud' 토픽 발행
        self.publisher = self.create_publisher(PointCloud2, '/pointcloud', 10)
        self.get_logger().info("RPLidar Node가 시작되었습니다.")

    def scan_callback(self, scan_data):
        try:
            # LaserScan 데이터를 PointCloud2로 변환
            cloud = self.laser_projector.projectLaser(scan_data)
            self.publisher.publish(cloud)
            self.get_logger().info("PointCloud2 메시지가 발행되었습니다.")
        except Exception as e:
            self.get_logger().error(f"LaserScan 데이터를 PointCloud2로 변환 중 오류 발생: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = RPLidarNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("RPLidar 노드 종료 중...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
