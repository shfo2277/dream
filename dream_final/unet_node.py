"""
unet_node – 노면 세그멘테이션

U-Net(unet129.h5, 입력 128x128)으로 22개 노면 클래스(보도/차도/점자블록/위험구역 등)를 예측한다.
결과 마스크를 10x10 격자로 줄여 RGB 값 JSON으로 가볍게 발행한다.
  sub: /camera/image_raw (sensor_msgs/Image)
  pub: /unet/predictions (std_msgs/String, JSON) {"mask": 10x10x[r, g, b]}
       /unet/segmented_image (sensor_msgs/Image) 원본 크기 컬러 마스크
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import tensorflow as tf
import numpy as np
import cv2
import json

class UNetNode(Node):
    def __init__(self):
        super().__init__('unet_node')

        # U-Net 모델 로드
        self.get_logger().info("Loading U-Net model...")
        model_path = '/home/d/d_ws/src/dr/unet129.h5'
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.get_logger().info("U-Net model loaded successfully.")

        # ROS 2 이미지 구독
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        # ROS 2 퍼블리셔 (예측 결과 전송)
        self.prediction_publisher = self.create_publisher(String, '/unet/predictions', 10)
        self.segmented_image_publisher = self.create_publisher(Image, '/unet/segmented_image', 10)

        # OpenCV와 ROS 2 간 변환 브릿지
        self.bridge = CvBridge()

        # 클래스 색상 매핑
        self.label_colors = {
            0: (0, 0, 0),              # 배경
            1: (0, 0, 255),            # sidewalk_blocks
            2: (217, 217, 217),        # sidewalk_cement
            3: (198, 89, 17),          # sidewalk_urethane
            4: (128, 128, 128),        # sidewalk_asphalt
            5: (255, 230, 153),        # sidewalk_soil_stone
            6: (55, 86, 35),           # sidewalk_damaged
            7: (110, 168, 70),         # sidewalk_other
            8: (255, 255, 0),          # braille_guide_blocks_normal
            9: (128, 96, 0),           # braille_guide_blocks_damaged
            10: (255, 128, 255),       # roadway_normal
            11: (255, 0, 255),         # roadway_crosswalk
            12: (230, 170, 255),       # alley_normal
            13: (208, 88, 255),        # alley_crosswalk
            14: (138, 60, 200),        # alley_speed_bump
            15: (88, 38, 128),         # alley_damaged
            16: (255, 155, 155),       # bike_lane_normal
            17: (255, 192, 0),         # caution_zone_stairs
            18: (255, 0, 0),           # caution_zone_manhole
            19: (0, 255, 0),           # caution_zone_tree_zone
            20: (255, 128, 0),         # caution_zone_grating
            21: (105, 105, 255),       # caution_zone_repair_zone
        }

    def image_callback(self, msg):
        """카메라 이미지를 받아 U-Net으로 처리하고 결과를 퍼블리시."""
        try:
            # ROS 2 이미지 메시지를 OpenCV 이미지로 변환
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # 이미지 크기 조정 (모델 입력 크기: 256x256)
            resized_frame = cv2.resize(frame, (128, 128))

            # 모델 입력으로 전처리
            input_data = resized_frame.astype(np.float32) / 255.0
            input_data = np.expand_dims(input_data, axis=0)  # 배치 차원 추가

            # 모델 예측 수행
            predictions = self.model.predict(input_data)

            # 예측된 마스크에서 각 픽셀의 클래스를 추출
            predicted_mask = np.argmax(predictions[0], axis=-1)

            # 예측된 마스크를 10x10 크기로 리사이즈
            small_mask = cv2.resize(predicted_mask.astype(np.uint8), (10, 10), interpolation=cv2.INTER_NEAREST)

            # 작은 마스크를 RGB 값으로 변환
            rgb_small_mask = np.zeros((10, 10, 3), dtype=np.uint8)
            for label, color in self.label_colors.items():
                rgb_small_mask[small_mask == label] = color

            # RGB 데이터를 JSON으로 변환하여 퍼블리시
            prediction_data = {
                'mask': rgb_small_mask.tolist()  # 리스트로 변환
            }
            prediction_msg = String()
            prediction_msg.data = json.dumps(prediction_data)
            self.prediction_publisher.publish(prediction_msg)

            # RGB 마스크를 원본 이미지 크기로 리사이즈
            rgb_mask = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3), dtype=np.uint8)
            for label, color in self.label_colors.items():
                rgb_mask[predicted_mask == label] = color
            rgb_mask_resized = cv2.resize(rgb_mask, (frame.shape[1], frame.shape[0]))

            # RGB 마스크를 ROS Image 형식으로 변환하여 퍼블리시
            segmented_image_msg = self.bridge.cv2_to_imgmsg(rgb_mask_resized, encoding='rgb8')
            self.segmented_image_publisher.publish(segmented_image_msg)

            # 결과 시각화
            combined_frame = cv2.addWeighted(frame, 0.6, rgb_mask_resized, 0.4, 0)
            cv2.imshow('U-Net Prediction', combined_frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = UNetNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

