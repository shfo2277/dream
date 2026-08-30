import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force TensorFlow to use CPU

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# .h5 파일 경로 지정
model_path = r'C:\jeon\j\best33.h5'
#model_path = r'C:\jeon\j\925PM.h5'



# 모델을 옵티마이저 없이 불러오기
model = tf.keras.models.load_model(model_path, compile=False)

# 모델 다시 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 이미지 파일 경로 지정
image_path = r'C:\jeon\j\test image\test7.jpg'

# OpenCV를 사용하여 이미지 불러오기
image = cv2.imread(image_path)

# 이미지가 제대로 로드되지 않으면 종료
if image is None:
    print("Error: Could not load image.")
    exit()

# 이미지 크기 조정 (모델 입력 크기에 맞춰 256x256으로 조정)
resized_image = cv2.resize(image, (256, 256))

# 모델에 넣기 위해 전처리 (모델의 입력 형식에 맞춰야 함)
input_data = resized_image.astype(np.float32) / 255.0  # 픽셀 값을 0-1 사이로 정규화
input_data = np.expand_dims(input_data, axis=0)  # 배치 차원 추가 (1, height, width, channels)

# 모델을 사용하여 예측
predictions = model.predict(input_data)

# 예측된 마스크에서 각 픽셀의 클래스를 추출 (다중 클래스 세그멘테이션일 경우)
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

# 원본 이미지와 마스크를 시각화
plt.figure(figsize=(12, 6))

# 원본 이미지
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

# 색상으로 표시된 예측 마스크
plt.subplot(1, 2, 2)
plt.imshow(rgb_mask)
plt.title('Predicted Mask')

plt.show()
