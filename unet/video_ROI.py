import tensorflow as tf
import numpy as np
import cv2

# .h5 파일 경로 지정
model_path = '/home/rth/dr/unet/result/unet129.h5'

# 모델 불러오기 (옵티마이저 없이)
model = tf.keras.models.load_model(model_path, compile=False)

# 모델 다시 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 동영상 파일 경로 지정
video_path = '/home/rth/dr/unet/testimage/이탈영상.mp4'
output_path = '/home/rth/dr/unet/result/output5.mp4'

# OpenCV를 사용하여 동영상 불러오기
cap = cv2.VideoCapture(video_path)

# 비디오가 제대로 열리지 않으면 종료
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 비디오 저장 설정
fps = int(cap.get(cv2.CAP_PROP_FPS))  # FPS 가져오기
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 프레임 너비
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 프레임 높이
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 코덱 설정
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# ROI 설정 좌표 (x, y, 너비, 높이)
roi_x, roi_y = 0, 300  # ROI 시작점
roi_w, roi_h = frame_width, frame_height - roi_y  # ROI 크기

# 클래스별 RGB 색상 매핑
label_colors = {
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

# 각 프레임 처리
frame_skip = 5  # 프레임 스킵 간격
frame_count = 0  # 현재 프레임 카운터

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break

    frame_count += 1

    # 프레임 스킵
    if frame_count % frame_skip != 0:
        continue

    # ROI 추출
    roi_frame = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    # 이미지 크기 조정 (128x128)
    resized_frame = cv2.resize(roi_frame, (128, 128))

    # 모델 입력 준비
    input_data = resized_frame.astype(np.float32) / 255.0  # 정규화
    input_data = np.expand_dims(input_data, axis=0)  # 배치 차원 추가

    # 모델 예측
    predictions = model.predict(input_data)
    predicted_mask = np.argmax(predictions[0], axis=-1)  # 클래스별로 가장 높은 값을 선택

    # 예측된 마스크를 RGB 색상으로 변환
    rgb_mask = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3), dtype=np.uint8)
    for label, color in label_colors.items():
        rgb_mask[predicted_mask == label] = color

    # RGB 마스크를 ROI 크기로 리사이즈
    rgb_mask_resized = cv2.resize(rgb_mask, (roi_w, roi_h))

    # ROI 영역과 마스크를 합성
    combined_roi = cv2.addWeighted(roi_frame, 0.6, rgb_mask_resized, 0.4, 0)

    # 합성된 ROI를 원본 프레임에 반영
    frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = combined_roi

    # 결과 비디오 저장
    out.write(frame)

    # 결과 출력
    cv2.imshow('Combined', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()
