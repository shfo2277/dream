import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json


class SidewalkBoundaryNode(Node):
    def __init__(self):
        super().__init__('sidewalk_boundary_node')

        # Prediction 토픽 구독
        self.subscription = self.create_subscription(
            String, '/unet/predictions', self.prediction_callback, 10
        )

        # 퍼블리시 토픽
        self.boundary_publisher = self.create_publisher(String, '/sidewalk/boundary_status', 10)

        # 인도로 간주할 RGB 값 정의 (차도 제외)
        self.sidewalk_colors = [
            [128, 128, 128],       # sidewalk_asphalt
            [255, 230, 153],       # sidewalk_soil_stone
            [55, 86, 35],          # sidewalk_damaged
            [0, 255, 0],           # caution_zone_tree_zone
            [255, 255, 0],         # braille_guide_blocks_normal (점자 블록)
            [128, 96, 0],          # braille_guide_blocks_damaged (손상된 점자 블록)
        ]

    def prediction_callback(self, msg):
        try:
            # JSON 데이터 파싱
            prediction_data = json.loads(msg.data)  # 문자열 데이터를 JSON으로 변환
            mask = prediction_data.get("mask", [])

            if not mask or not isinstance(mask, list):  # 데이터 확인
                self.get_logger().error("Invalid mask data received.")
                return

            # 6, 7, 8, 9번째 행만 선택
            selected_rows = [mask[6], mask[7], mask[8], mask[9]]

            # 왼쪽과 오른쪽 각각 인도 여부를 계산
            left_count, right_count = 0, 0
            for row in selected_rows:
                if not isinstance(row, list):  # 행 데이터가 리스트인지 확인
                    self.get_logger().error(f"Invalid row data: {row}")
                    return

                # 왼쪽과 오른쪽으로 나눠서 인도 픽셀 계산
                half = len(row) // 2
                left_pixels = row[:half]
                right_pixels = row[half:]

                # 왼쪽, 오른쪽 각각 인도 픽셀 비율 계산
                left_sidewalk_ratio = sum(1 for pixel in left_pixels if pixel in self.sidewalk_colors) / len(left_pixels)
                right_sidewalk_ratio = sum(1 for pixel in right_pixels if pixel in self.sidewalk_colors) / len(right_pixels)

                # 디버깅: 각 비율 출력
                self.get_logger().info(f"Left sidewalk ratio: {left_sidewalk_ratio*100:.2f}%")
                self.get_logger().info(f"Right sidewalk ratio: {right_sidewalk_ratio*100:.2f}%")

                if left_sidewalk_ratio >= 0.4: 
                    left_count += 1
                if right_sidewalk_ratio >= 0.4:
                    right_count += 1
                    

            if left_count == 0 and right_count == 0:
                status = "on"  # 인도 위 
                direction = "center"  # 경계가 없으므로 방향은 알 수 없음

            elif left_count > 0 and right_count == 0:
                status = "on"  # 왼쪽 차도, 오른쪽 인도 
                direction = "left"  # 인도인데 왼쪽으로 이탈 (왼쪽에 차도)

            elif right_count > 0 and left_count == 0:
                status = "on"  # 오른쪽 차도, 왼쪽 인도 
                direction = "right"  # 오른쪽에 차도

            elif left_count > 0 and right_count > 0:
                status = "off"  # 양쪽 모두 차도 위
                direction = "unknown"  # 양쪽 모두 
                    



            # 결과 퍼블리시
            boundary_status = {
                "status": status,
                "direction": direction,
            }

            boundary_msg = String()
            boundary_msg.data = json.dumps(boundary_status)
            self.boundary_publisher.publish(boundary_msg)

        except Exception as e:
            self.get_logger().error(f"Failed to process prediction: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = SidewalkBoundaryNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
