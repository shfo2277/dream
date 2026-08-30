import cv2

def main():
    # 카메라 열기 (기본 카메라: 0번 장치)
    cap = cv2.VideoCapture(0)  #주의 수정하기, 연결된 카메라 장치 확인: ls /dev/video*


    # 카메라가 열렸는지 확인
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    while True:
        # 카메라에서 프레임 읽기
        ret, frame = cap.read()

        # 프레임 읽기 실패 시
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # 화면에 프레임 표시
        cv2.imshow('Camera Feed', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

