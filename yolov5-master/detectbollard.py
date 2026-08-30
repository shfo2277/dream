import torch
import cv2
import time

# YOLO 모델 파일 로드
model_path = 'best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# 웹캠 초기화 (카메라 인덱스 0)
cap = cv2.VideoCapture(0)

# 카메라 연결 확인
if not cap.isOpened():
    print("카메라가 연결되지 않았습니다.")
    exit()

# FPS 측정을 위한 초기화
prev_time = 0

# Confidence threshold 설정
confidence_threshold = 0.8

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # YOLO 모델로 객체 탐지 수행
    results = model(frame)

    # FPS 계산
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # 탐지 결과 필터링 (confidence threshold 적용)
    filtered_results = []
    for det in results.xyxy[0]:  # 각 탐지된 객체에 대해
        x1, y1, x2, y2, conf, cls = det  # 좌표, confidence, 클래스 ID 추출
        if conf >= confidence_threshold:  # confidence가 threshold 이상인 경우
            filtered_results.append([x1, y1, x2, y2, conf, cls])

    # 결과 이미지 그리기
    output_frame = frame.copy()  # 원본 프레임을 복사하여 결과 표시

    # confidence가 threshold 이상인 객체만 그리기
    for det in filtered_results:
        x1, y1, x2, y2, conf, cls = map(int, det)
        label = f'bollard'
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Bounding box 그리기
        cv2.putText(output_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Confidence 표시

    # FPS 텍스트 추가
    cv2.putText(output_frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 결과 표시
    cv2.imshow('YOLOv5 Detection', output_frame)

    # ESC 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 웹캠 및 창 닫기
cap.release()
cv2.destroyAllWindows()
