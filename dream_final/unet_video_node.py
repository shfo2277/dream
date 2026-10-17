import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
import tensorflow as tf
import numpy as np
import cv2
import json
from cv_bridge import CvBridge


class VideoSegmentationNode(Node):
    def __init__(self):
        super().__init__('video_segmentation_node')

        # 모델 로드
        self.get_logger().info("Loading U-Net model...")
        model_path = r'/home/d/d_ws/src/dr/unet129.h5'
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.get_logger().info("Model loaded successfully.")

        # 동영상 파일 경로
        self.video_path = r'/home/d/d_ws/src/dr/5.mp4'

        # OpenCV와 ROS 2 변환 브릿지
        self.bridge = CvBridge()

        # 10x10 행렬 퍼블리셔
        self.matrix_publisher = self.create_publisher(String, '/unet/predictions', 10)

        # 비디오 처리
        self.process_video()

    def process_video(self):
        """동영상 프레임을 처리하고 결과를 퍼블리시"""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.get_logger().error("Could not open video.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 이미지 크기 조정 (모델 입력 크기에 맞춰 256x256으로 조정)
            resized_frame = cv2.resize(frame, (128, 128))

            # 모델 입력 전처리
            input_data = resized_frame.astype(np.float32) / 255.0
            input_data = np.expand_dims(input_data, axis=0)

            # 예측
            predictions = self.model.predict(input_data)
            predicted_mask = np.argmax(predictions[0], axis=-1)

            # 10x10 크기로 축소
            small_mask = cv2.resize(predicted_mask.astype(np.uint8), (10, 10), interpolation=cv2.INTER_NEAREST)

            # RGB 변환
            rgb_small_mask = self._convert_to_rgb(small_mask)

            # JSON 메시지 생성
            rgb_data_list = rgb_small_mask.tolist()
            matrix_msg = String()
            matrix_msg.data = json.dumps({"mask": rgb_data_list})
            self.matrix_publisher.publish(matrix_msg)

            # 결과 시각화
            combined_frame = self._visualize(frame, predicted_mask)
            cv2.imshow('Segmentation', combined_frame)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def _convert_to_rgb(self, mask):
        """10x10 행렬을 RGB로 변환"""
        label_colors = {
            0: (0, 0, 0),
            1: (0, 0, 255),
            2: (217, 217, 217),
            3: (198, 89, 17),
            4: (128, 128, 128),
            5: (255, 230, 153),
            6: (55, 86, 35),
            7: (110, 168, 70),
            8: (255, 255, 0),
            9: (128, 96, 0),
            10: (255, 128, 255),
            11: (255, 0, 255),
            12: (230, 170, 255),
            13: (208, 88, 255),
            14: (138, 60, 200),
            15: (88, 38, 128),
            16: (255, 155, 155),
            17: (255, 192, 0),
            18: (255, 0, 0),
            19: (0, 255, 0),
            20: (255, 128, 0),
            21: (105, 105, 255),
        }
        # 각 픽셀을 RGB 값으로 변환
        rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for label, color in label_colors.items():
            rgb_mask[mask == label] = color
        return rgb_mask

    def _visualize(self, frame, mask):
        """예측 결과를 원본 이미지 위에 시각화"""
        rgb_mask = self._convert_to_rgb(mask)
        rgb_mask_resized = cv2.resize(rgb_mask, (frame.shape[1], frame.shape[0]))
        combined_frame = cv2.addWeighted(frame, 0.6, rgb_mask_resized, 0.4, 0)
        return combined_frame


def main(args=None):
    rclpy.init(args=args)
    node = VideoSegmentationNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
