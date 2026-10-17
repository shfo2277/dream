"""
websocket1 – 장애물 정보 전달

퓨전 결과를 WebSocket(rosbridge) 연동용 토픽으로 그대로 재발행한다.
  sub: /fusion_results
  pub: /websocket1
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json  # JSON 메시지 처리용

class FusionResultsProcessor(Node):
    def __init__(self):
        super().__init__('fusion_results_processor')
        
        # 퍼블리셔 생성: /websocket
        self.result_pub = self.create_publisher(String, '/websocket1', 10)
        
        # 구독자 생성: /fusion_results
        self.result_sub = self.create_subscription(
            String,
            '/fusion_results',
            self.fusion_result_callback,
            10
        )
        self.get_logger().info("Fusion Results Processor가 시작되었습니다.")

    def fusion_result_callback(self, msg):
        try:
            # 받은 메시지 데이터 처리 (이미 JSON 형식으로 예상)
            received_data = msg.data
            
            # 받은 데이터를 그대로 퍼블리시
            output_msg = String()
            output_msg.data = received_data  # 받은 데이터를 그대로 사용
            self.result_pub.publish(output_msg)
            
            # 로그 출력
            self.get_logger().info(f"받은 데이터: {received_data}")
            self.get_logger().info(f"퍼블리시: {output_msg.data}")
        
        except Exception as e:
            self.get_logger().error(f"데이터 처리 중 오류 발생: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    fusion_results_processor = FusionResultsProcessor()
    rclpy.spin(fusion_results_processor)
    fusion_results_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

