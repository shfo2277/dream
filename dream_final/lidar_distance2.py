import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

class ObstacleDistanceNode(Node):
    def __init__(self):
        super().__init__('obstacle_distance_node')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        self.subscription  # prevent unused variable warning

    def scan_callback(self, msg):
        # LaserScan 메시지에서 거리 정보를 가져옵니다.
        distances = msg.ranges
        angle_min = msg.angle_min  # 스캔의 시작 각도
        angle_increment = msg.angle_increment  # 각도 증가값 (라이다 분해능)

        # inf 값을 제외한 유효한 거리 리스트 생성
        valid_distances_with_index = [
            (index, distance) for index, distance in enumerate(distances) if distance < float('inf')
        ]

        if valid_distances_with_index:
            # 가장 가까운 장애물 계산
            min_index, min_distance = min(valid_distances_with_index, key=lambda x: x[1])
            min_angle = angle_min + (min_index * angle_increment)  # 인덱스를 각도로 변환
            min_angle_deg = min_angle * (180.0 / 3.141592653589793)  # 라디안을 각도로 변환

            self.get_logger().info(
                f'가장 가까운 장애물 거리: {min_distance:.2f} m, 각도: {min_angle_deg:.2f}°'
            )
        else:
            # 유효한 장애물이 없는 경우
            self.get_logger().info('감지된 장애물이 없습니다.')

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDistanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

