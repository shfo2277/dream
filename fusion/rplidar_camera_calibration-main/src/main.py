#!/usr/bin/env python3
# -- coding: utf-8 --
import cv2
import rospy
import numpy as np
from lidar import LaserToPointCloud
import sensor_msgs.point_cloud2 as pc2

# 카메라 내부 및 외부 파라미터
INTRINSIC = np.array([
    [390.20006229, 0.0, 321.54814476],
    [0.0, 390.02222138, 238.40380784],
    [0.0, 0.0, 1.0]
])

EXTRINSIC_R = np.array([
    [0.12355993, -0.99161296, -0.03790349],
    [-0.19535472, 0.01314206, -0.98064459],
    [0.97291802, 0.128573, -0.19209244]
])

EXTRINSIC_T = np.array([[-0.0013882], [0.42830764], [-0.34803576]])


def transform_lidar_to_camera(points):
    """
    LiDAR 데이터를 카메라 좌표계로 변환합니다.
    :param points: LiDAR 포인트 (Nx3)
    :return: 변환된 카메라 좌표 (Nx3)
    """
    points = np.array(points, dtype=np.float32).T  # (3xN)
    transformed_points = EXTRINSIC_R @ points + EXTRINSIC_T  # R * P + T
    return transformed_points.T  # (Nx3)


def project_to_image(points):
    """
    카메라 3D 좌표를 이미지 2D 좌표로 변환합니다.
    :param points: 카메라 좌표 (Nx3)
    :return: 이미지 좌표 (Nx2)
    """
    points_2d = INTRINSIC @ points.T  # (3x3) * (3xN) = (3xN)
    points_2d = points_2d[:2] / points_2d[2]  # Normalize
    return points_2d.T  # (Nx2)


def main():
    rospy.init_node('sensor_fusion_node', anonymous=True)

    # Laser to PointCloud 노드 초기화
    ltp = LaserToPointCloud()

    # 카메라 열기
    cap = cv2.VideoCapture(4)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    while not rospy.is_shutdown():
        # 카메라 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("카메라 프레임을 읽을 수 없습니다.")
            break

        # LiDAR 데이터 읽기
        lidar_points = []
        try:
            lidar_cloud = rospy.wait_for_message('/pointcloud', PointCloud2, timeout=1.0)
            for point in pc2.read_points(lidar_cloud, skip_nans=True):
                lidar_points.append([point[0], point[1], point[2]])  # x, y, z
        except rospy.ROSException:
            print("LiDAR 데이터를 읽을 수 없습니다.")
            continue

        if len(lidar_points) == 0:
            print("LiDAR 데이터가 비어 있습니다.")
            continue

        # LiDAR -> 카메라 좌표 변환
        camera_points = transform_lidar_to_camera(lidar_points)

        # 카메라 좌표 -> 이미지 좌표 변환
        image_points = project_to_image(camera_points)

        # 이미지에 LiDAR 점 표시
        for pt in image_points:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:  # 이미지 범위 내 점만 표시
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

        # 프레임 표시
        cv2.imshow('Sensor Fusion', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
