#!/usr/bin/env python
# -- coding: utf-8 --

import numpy as np
import cv2
import os
import glob


class CameraCali:
    def __init__(self):
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.wc = 8  # the number of horizontal checkerboard patterns - 1
        self.hc = 6  # the number of vertical checkerboard patterns - 1
        self.objp = np.zeros((self.wc * self.hc, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.wc, 0:self.hc].T.reshape(-1, 2)

        self.objpoints = []
        self.imgpoints = []

    def calibrate(self):
        # import images in directory
        images = glob.glob(os.path.dirname(__file__) +
                           '/checkerboard_jpg/*.jpg')

        for frame in images:
            img = cv2.imread(frame)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # change to gray scale

            # find the checkerboard
            ret, corners = cv2.findChessboardCorners(
                gray, (self.wc, self.hc),
                cv2.CALIB_CB_ADAPTIVE_THRESH +
                cv2.CALIB_CB_FAST_CHECK +
                cv2.CALIB_CB_NORMALIZE_IMAGE
            )  # 체스 보드 찾기

            # if ret is False, please check your checkerboard (wc, hc)
            print(ret, end=' ')

            if ret:
                self.objpoints.append(self.objp)

                corners2 = cv2.cornerSubPix(
                    gray, corners, (10, 10), (-1, -1), self.criteria)
                self.imgpoints.append(corners2)

                # draw images using corner points
                img = cv2.drawChessboardCorners(img, (self.wc, self.hc), corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey()

        # Camera calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, gray.shape[::-1], None, None)

        # Set camera matrix and distortion coefficients
        self.cameraMatrix = mtx
        self.dist_coeffs = dist

        print("Calibration completed successfully.")

    def solve_pnp(self):
        # Example 2D-image and 3D-world coordinates
        points_2D = np.array([
            (454, 245), (246, 252), (226, 305), (193, 225),
            (312, 335), (341, 355), (366, 439), (306, 241),
            (582, 338), (68, 364)
        ], dtype="double")

        points_3D = np.array([
            (1.61, -0.338, 0.00374), (1.69, 0.19, 0.00616),
            (1.39, 0.177, 0.00691), (1.82, 0.376, 0.00136),
            (1.3, -0.0179, 0.00335), (1.19, -0.0499, 0.00132),
            (0.826, -0.104, 0.00368), (1.94, -0.0139, 0.00546),
            (1.28, -0.615, 0.00479), (1.13, 0.442, 0.000586)
        ], dtype="double")

        retval, rvec, tvec = cv2.solvePnP(
            points_3D, points_2D, self.cameraMatrix,
            self.dist_coeffs, rvec=None, tvec=None,
            useExtrinsicGuess=None, flags=None
        )

        # Convert rotation vector to matrix
        rvec_matrix, _ = cv2.Rodrigues(rvec)

        # Homogeneous coordinate matrix
        intrinsic = np.append(self.cameraMatrix, [[0, 0, 0]], axis=0)
        intrinsic = np.append(intrinsic, [[0], [0], [0], [1]], axis=1)

        extrinsic = np.append(rvec_matrix, tvec, axis=1)
        extrinsic = np.append(extrinsic, [[0, 0, 0, 1]], axis=0)

        print("\nIntrinsic matrix:")
        print(intrinsic)
        print("\nExtrinsic matrix:")
        print(extrinsic)
        print("\nRotation matrix:")
        print(rvec_matrix)
        print("\nTranslation vector:")
        print(tvec)


if __name__ == '__main__':
    cam_cali = CameraCali()
    cam_cali.calibrate()
    cam_cali.solve_pnp()
