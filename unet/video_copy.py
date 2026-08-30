import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# .h5 파일 경로 지정
model_path = '/home/rth/dr/unet/unetsc.h5'

# 모델을 옵티마이저 없이 불러오기
model = tf.keras.models.load_model(model_path, compile=False)

# 모델 다시 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 동영상 파일 경로 지정
video_path = '/home/rth/dr/unet/testimage/konkuk_road2.mp4'
output_path = '/home/rth/dr/unet/result.5mp4'

# OpenCV를 사용하여 동영상 불러오기
cap = cv2.VideoCapture(video_path)

# 동영상이 제대로 열리지 않으면 종료
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# 전체 프레임 수 확인
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames: {total_frames}")

# 비디오 저장 객체 생성
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 현재 프레임 번호 초기화
current_frame = 0

# 각 프레임을 처리
while True:
    ret, frame = cap.read()
    
    # 프레임이 제대로 읽히지 않으면 종료
    if not ret:
        break

    # 현재 프레임 번호 업데이트
    current_frame += 1
    print(f"Processing frame {current_frame}/{total_frames}")

    # 이미지 크기 조정 (모델 입력 크기에 맞춰 256x256으로 조정)
    resized_frame = cv2.resize(frame, (256, 256))

    # 모델에 넣기 위해 전처리 (모델의 입력 형식에 맞춰야 함)
    input_data = resized_frame.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)

    # 모델을 사용하여 예측
    predictions = model.predict(input_data)
    predicted_mask = np.argmax(predictions[0], axis=-1)

    # 각 클래스에 대응하는 RGB 색상 매핑
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

    # 예측된 마스크를 RGB 색상으로 변환
    rgb_mask = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3), dtype=np.uint8)
    for label, color in label_colors.items():
        rgb_mask[predicted_mask == label] = color

    # rgb_mask를 원본 프레임 크기로 리사이즈
    rgb_mask_resized = cv2.resize(rgb_mask, (frame.shape[1], frame.shape[0]))

    # 원본 이미지와 마스크를 합칩니다.
    combined = cv2.addWeighted(frame, 0.6, rgb_mask_resized, 0.4, 0)
   
    # 결과 저장/ 비디오 저장 코드
    out.write(combined)

    # 결과 출력
    cv2.imshow('Combined', combined)

    # 'q' 키를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()
