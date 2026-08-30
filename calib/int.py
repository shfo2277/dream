import cv2
import numpy as np
import glob
import os

# === 1. 설정 ===
chessboard_size = (9, 7)  # 체스보드의 내부 코너 수 (가로, 세로)
square_size = 1.76  # 체스보드 한 칸의 크기 (cm)
image_path = '/home/rth/calib/image/*.jpg'  # 캘리브레이션 이미지 경로
corner_output_path = '/home/rth/calib/corner_images/'  # 코너 감지 결과 저장 경로
undistorted_output_path = '/home/rth/calib/undistorted_images/'  # 왜곡 보정 결과 저장 경로

# 디렉토리 생성
os.makedirs(corner_output_path, exist_ok=True)
os.makedirs(undistorted_output_path, exist_ok=True)

# === 2. 체스보드 3D 점 준비 ===
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D 체스보드 점
imgpoints = []  # 2D 이미지 평면 점

# === 3. 체스보드 이미지 읽기 및 코너 감지 ===
images = glob.glob(image_path)
print(f"Number of images found: {len(images)}")

if len(images) == 0:
    print("Error: No images found. Check the image directory path.")
    exit()

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 체스보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(
        gray, chessboard_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret:
        print(f"Chessboard corners detected in {fname}")
        objpoints.append(objp)
        imgpoints.append(corners)

        # 코너 표시 및 저장
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        output_filename = os.path.join(corner_output_path, f"corners_{os.path.basename(fname)}")
        cv2.imwrite(output_filename, img)
        print(f"Saved corner detection result: {output_filename}")
    else:
        print(f"Failed to detect chessboard corners in {fname}")

cv2.destroyAllWindows()

if len(objpoints) == 0 or len(imgpoints) == 0:
    print("Error: No corners detected in any image. Check the chessboard pattern and settings.")
    exit()

# === 4. 카메라 캘리브레이션 ===
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# 내부 파라미터 출력
print("\n=== Calibration Results ===")
print("Camera Matrix (Intrinsic Parameters):")
print(camera_matrix)
print("\nDistortion Coefficients:")
print(dist_coeffs)

# 내부 파라미터 저장
np.save("/home/rth/calib/camera_matrix.npy", camera_matrix)
np.save("/home/rth/calib/dist_coeffs.npy", dist_coeffs)
print("\nCamera Matrix and Distortion Coefficients saved as .npy files.")

# === 5. 왜곡 보정 ===
for fname in images:
    img = cv2.imread(fname)
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    # 왜곡 보정
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # ROI 적용 (유효 영역만 잘라내기)
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y+h, x:x+w]

    # 보정된 이미지 저장
    output_filename = os.path.join(undistorted_output_path, f"undistorted_{os.path.basename(fname)}")
    cv2.imwrite(output_filename, undistorted_img)
    print(f"Saved undistorted image: {output_filename}")

