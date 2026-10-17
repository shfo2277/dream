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

    	# 특정 범위의 거리 정보만 출력
    	front_distances = distances[:32] + distances[-32:]

    	# inf 값을 제외한 유효한 거리 리스트 생성
    	valid_distances = [d for d in front_distances if d < float('inf')]

    	if valid_distances:
        	# 장애물까지의 가장 가까운 거리 계산
        	min_distance = min(valid_distances)
        	self.get_logger().info(f'가장 가까운 장애물 거리: {min_distance:.2f} m')
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

