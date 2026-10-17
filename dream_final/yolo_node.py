#쓰레딩 추가
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import torch
import time
import json
import threading
import queue


class YOLOv5Node(Node):
    def __init__(self):
        super().__init__('yolov5')

        # YOLO 모델 로드
        self.get_logger().info("Loading YOLOv5 model...")
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/d/d_ws/src/dr/best.pt', force_reload=True)
        self.get_logger().info("YOLOv5 model loaded successfully.")

        # 카메라 이미지 구독
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        # 바운딩박스 결과 게시
        self.publisher = self.create_publisher(String, '/yolov5/detections', 10)

        # OpenCV 및 FPS 계산을 위한 초기화
        self.bridge = CvBridge()
        self.confidence_threshold = 0.8  # Confidence threshold 설정

        # 큐와 스레드 초기화
        self.frame_queue = queue.Queue(maxsize=5)
        self.detection_thread = threading.Thread(target=self.detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()

        # FPS 측정을 위한 변수
        self.prev_time = 0

    def image_callback(self, msg):
        """카메라 이미지 수신 콜백."""
        # ROS2 이미지 메시지를 OpenCV 이미지로 변환
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 큐가 꽉 차지 않았으면 프레임 추가
        if not self.frame_queue.full():
            self.frame_queue.put(frame)

    def detection_loop(self):
        """YOLOv5 탐지 루프 (별도 스레드에서 실행)."""
        while rclpy.ok():
            try:
                # 큐에서 프레임 가져오기
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            # YOLO 모델로 객체 탐지 수행
            results = self.model(frame)

            # FPS 계산
            curr_time = time.time()
            fps = 1 / (curr_time - self.prev_time) if self.prev_time > 0 else 0
            self.prev_time = curr_time

            # 탐지 결과 필터링 및 게시 데이터 생성
            detections = []
            for det in results.xyxy[0].cpu().numpy():
                x1, y1, x2, y2, conf, cls = det
                if conf >= self.confidence_threshold:
                    class_name = self.model.names[int(cls)]  # 클래스 이름 가져오기
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class_name': class_name  # 클래스 이름 추가
                    })

            # 탐지 결과 게시 (JSON 형식)
            if detections:
                msg = String()
                msg.data = json.dumps({'detections': detections})
                self.publisher.publish(msg)

            # 결과를 이미지에 표시
            output_frame = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                conf = det['confidence']
                class_name = det['class_name']
                label = f'{class_name}: {conf:.2f}'
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(output_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # FPS 표시
            cv2.putText(output_frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # 결과 이미지 출력
            cv2.imshow('YOLOv5 Detection', output_frame)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = YOLOv5Node()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

