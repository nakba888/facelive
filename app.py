import os
import sys
import time
import urllib.request
import cv2
import numpy as np

MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
MODEL_PATH = "face_detection_yunet_2023mar.onnx"


def download_model_if_needed():
    """YuNet ONNX 모델 파일이 없으면 자동으로 다운로드합니다."""
    if not os.path.exists(MODEL_PATH):
        print(f"[Info] Downloading YuNet ONNX model from {MODEL_URL} ...")
        try:
            # User-Agent 헤더 추가하여 다운로드
            req = urllib.request.Request(
                MODEL_URL,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req) as response, open(MODEL_PATH, 'wb') as out_file:
                out_file.write(response.read())
            print("[Info] YuNet model download complete!")
        except Exception as e:
            print(f"[Error] Failed to download YuNet model: {e}")
            sys.exit(1)


# 카메라 해상도 프리셋 정의 (1번: 가장 낮음 -> 4번: 가장 높음)
RESOLUTIONS = [
    ("240p QVGA", 320, 240),   # 1번: 낮음
    ("480p SD", 640, 480),     # 2번: 보통
    ("720p HD", 1280, 720),    # 3번: 높음
    ("1080p FHD", 1920, 1080), # 4번: 최고 높음
]


def set_camera_resolution(cap, detector, width, height, current_w, current_h):
    """카메라 해상도를 안정적으로 설정하고, 미지원 해상도일 경우 이전 해상도로 자동 복구합니다."""
    # 1. 새 해상도 설정 시도
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # 연속 프레임 읽기로 카메라 버퍼 비우기 및 프레임 유효성 검사
    frame = None
    ret = False
    for _ in range(5):
        try:
            ret, frame = cap.read()
            if ret and frame is not None and hasattr(frame, 'size') and frame.size > 0:
                break
        except Exception:
            ret = False
            frame = None

    if ret and frame is not None and hasattr(frame, 'size') and frame.size > 0:
        actual_h, actual_w, _ = frame.shape
        detector.setInputSize((actual_w, actual_h))
        return actual_w, actual_h, frame
    else:
        # 카메라가 해당 해상도를 지원하지 않는 경우 이전 해상도로 복구
        print(f"[알림] 웹캠이 {width}x{height} 해상도를 지원하지 않습니다. 이전 해상도({current_w}x{current_h})로 복구합니다.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, current_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, current_h)

        backup_frame = None
        for _ in range(5):
            try:
                ret_b, f_b = cap.read()
                if ret_b and f_b is not None and hasattr(f_b, 'size') and f_b.size > 0:
                    backup_frame = f_b
                    break
            except Exception:
                pass

        if backup_frame is not None:
            actual_h, actual_w, _ = backup_frame.shape
            detector.setInputSize((actual_w, actual_h))
            return actual_w, actual_h, backup_frame

        return current_w, current_h, None


# 무지개 색상 정의 (BGR 포맷): 빨, 주, 노, 초, 파, 남, 보
RAINBOW_COLORS = [
    (0, 0, 255),      # #1위 (가장 큼): 빨강 (Red)
    (0, 165, 255),    # #2위: 주황 (Orange)
    (0, 255, 255),    # #3위: 노랑 (Yellow)
    (0, 255, 0),      # #4위: 초록 (Green)
    (255, 100, 0),    # #5위: 파랑 (Blue)
    (130, 0, 75),     # #6위: 남색 (Indigo)
    (211, 0, 148),    # #7위: 보라 (Violet)
]


def draw_faces(image, faces, score_threshold=0.8):
    """감지된 얼굴을 크기(면적) 순으로 정렬하여 무지개 색상(빨주노초파남보)의 #1, #2, #3 순위와 랜드마크를 시각화합니다."""
    if faces is None:
        return 0

    # 신뢰도 임계값을 넘는 얼굴만 필터링
    valid_faces = [f for f in faces if f[-1] >= score_threshold]
    if not valid_faces:
        return 0

    # 얼굴 바운딩 박스 면적(width * height) 기준 내림차순 정렬 (가장 큰 얼굴이 #1)
    valid_faces.sort(key=lambda f: f[2] * f[3], reverse=True)

    for rank, face in enumerate(valid_faces, start=1):
        score = face[-1]
        bbox = list(map(int, face[:4]))
        x, y, w, h = bbox

        # 순위에 따른 무지개 색상 지정 (1위: 빨강, 2위: 주황, 3위: 노랑...)
        box_color = RAINBOW_COLORS[(rank - 1) % len(RAINBOW_COLORS)]

        # 얼굴 영역 사각형 그리기 (무지개 색상 적용)
        cv2.rectangle(image, (x, y), (x + w, y + h), box_color, 2)

        # 상단 라벨 (#1 | Score: 0.95)
        label = f"#{rank} (Score: {score:.2f})"

        # 라벨 배경 상자 그리기 (글자 읽기 쉽게)
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = max(y - 5, text_h + 5)
        cv2.rectangle(
            image,
            (x, label_y - text_h - 4),
            (x + text_w + 6, label_y + baseline),
            (0, 0, 0), -1
        )
        cv2.putText(
            image, label, (x + 3, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2, cv2.LINE_AA
        )

        # 얼굴 상자 내부 좌측 상단에 큰 순위 숫자 표시 (#1, #2...)
        rank_str = f"#{rank}"
        cv2.putText(
            image, rank_str, (x + 8, y + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2, cv2.LINE_AA
        )

        # 5개 랜드마크 (오른쪽 눈, 왼쪽 눈, 코끝, 오른쪽 입꼬리, 왼쪽 입꼬리)
        landmarks = list(map(int, face[4:14]))
        colors = [
            (255, 0, 0),    # 오른쪽 눈 (파랑)
            (0, 0, 255),    # 왼쪽 눈 (빨강)
            (0, 255, 255),  # 코끝 (노랑)
            (255, 255, 0),  # 오른쪽 입꼬리 (청록)
            (255, 0, 255)   # 왼쪽 입꼬리 (보라)
        ]
        for i in range(5):
            lx, ly = landmarks[i * 2], landmarks[i * 2 + 1]
            cv2.circle(image, (lx, ly), 4, colors[i], -1)

    return len(valid_faces)



def main():
    # 1. YuNet 모델 다운로드 검사
    download_model_if_needed()

    # 2. 웹캠 연결 (Windows DSHOW 백엔드 우선 적용으로 드라이버 안정성 향상)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[Error] 카메라는 연결할 수 없습니다. (카메라 장치 번호를 확인해주세요)")
        return

    # 첫 프레임 읽어 입력 크기 설정
    ret, frame = False, None
    for _ in range(5):
        ret, frame = cap.read()
        if ret and frame is not None:
            break

    if not ret or frame is None:
        print("[Error] 카메라 프레임을 읽어올 수 없습니다.")
        return

    h, w, _ = frame.shape

    # 3. OpenCV YuNet 얼굴 검출기 초기화
    try:
        detector = cv2.FaceDetectorYN.create(
            model=MODEL_PATH,
            config="",
            input_size=(w, h),
            score_threshold=0.8,
            nms_threshold=0.3,
            top_k=5000
        )
    except Exception as e:
        print(f"[Error] FaceDetectorYN 생성 실패: {e}")
        return

    is_frozen = False
    res_index = 2  # 기본값: 3번 720p HD
    show_fps = True # F 키로 FPS 표시 토글
    prev_time = time.time()
    fps = 0.0

    raw_frame = frame
    detected_faces = None
    face_count = 0

    print("\n" + "=" * 65)
    print("      YuNet OpenCV 얼굴 감지 & 실시간 화질/FPS 조절 (facelive)")
    print("=" * 65)
    print(" 조작 방법:")
    print("   [Space Bar]   : 화면 정지(Pause) / 다시 재생(Resume)")
    print("   [1 / 2 / 3 / 4]: 화질(해상도) 즉시 변경 (1:낮음 -> 4:높음)")
    print("                     1: 240p | 2: 480p | 3: 720p(HD) | 4: 1080p(FHD)")
    print("   [R]           : 다음 화질로 순환 변경")
    print("   [F]           : 실시간 FPS(프레임 수) 표시 ON / OFF")
    print("   [S]           : 현재 화면 이미지 파일로 저장")
    print("   [Q / ESC]     : 프로그램 종료")
    print("=" * 65 + "\n")

    while True:
        # FPS 측정 (실시간 계산)
        curr_time = time.time()
        time_diff = curr_time - prev_time
        prev_time = curr_time
        if time_diff > 0:
            inst_fps = 1.0 / time_diff
            fps = inst_fps if fps == 0.0 else 0.9 * fps + 0.1 * inst_fps

        # 화면 재생 상태(not is_frozen)일 때만 카메라인 새 프레임과 얼굴 감지 업데이트
        if not is_frozen:
            try:
                ret, frame = cap.read()
            except Exception:
                ret = False

            if not ret or frame is None or not hasattr(frame, 'size') or frame.size == 0:
                continue

            # 해상도가 변경된 경우 입력 크기 업데이트
            if (frame.shape[1], frame.shape[0]) != (w, h):
                w, h = frame.shape[1], frame.shape[0]
                detector.setInputSize((w, h))

            raw_frame = frame.copy()
            # YuNet 얼굴 감지 수행
            _, detected_faces = detector.detect(raw_frame)

        # -------------------------------------------------------------
        # 매 루프마다 raw_frame에서 새로 복사하여 시각화 (텍스트 겹침 방지)
        # -------------------------------------------------------------
        display_frame = raw_frame.copy()

        # 1. 얼굴 및 순위(#1, #2...) 그리기
        face_count = draw_faces(display_frame, detected_faces, score_threshold=0.8)

        # 2. 상단 좌측: 상태 표시 (LIVE는 녹색, FROZEN은 빨간색으로 깔끔하게 1개만 표시)
        if not is_frozen:
            cv2.circle(display_frame, (25, 25), 8, (0, 255, 0), -1)
            cv2.putText(
                display_frame, f"LIVE | Faces: {face_count}", (42, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA
            )
        else:
            cv2.circle(display_frame, (25, 25), 8, (0, 0, 255), -1)
            cv2.putText(
                display_frame, f"FROZEN (PAUSED) | Faces: {face_count}", (42, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA
            )

        # 3. 상단 우측: 현재 해상도 배지 표시 (예: Res: 1280x720 (720p HD))
        res_name = RESOLUTIONS[res_index][0]
        res_info = f"Res: {w}x{h} ({res_name})"
        (res_w, res_h), _ = cv2.getTextSize(res_info, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(
            display_frame,
            (w - res_w - 20, 10),
            (w - 10, 40),
            (40, 40, 40), -1
        )
        cv2.putText(
            display_frame, res_info, (w - res_w - 15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2, cv2.LINE_AA
        )

        # 4. 상단 우측: FPS 표시 배지 (F키로 토글 가능)
        if show_fps:
            fps_info = f"FPS: {fps:.1f}"
            (fps_w, _), _ = cv2.getTextSize(fps_info, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            fps_x = w - res_w - fps_w - 45
            cv2.rectangle(
                display_frame,
                (fps_x - 10, 10),
                (fps_x + fps_w + 10, 40),
                (40, 40, 40), -1
            )
            cv2.putText(
                display_frame, fps_info, (fps_x, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA
            )

        # 5. 하단 조작 가이드 안내
        guide_text = "[Space]: Freeze  |  [1-4/R]: Res (1:Low->4:High)  |  [F]: FPS  |  [S]: Save  |  [Q]: Quit"
        cv2.putText(
            display_frame, guide_text, (15, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA
        )

        # 화면 출력
        cv2.imshow("YuNet Face Detection - facelive", display_frame)



        # 키 입력 대기 (1ms)
        key = cv2.waitKey(1) & 0xFF

        # Q 키 또는 ESC 키: 종료
        if key in (27, ord('q'), ord('Q')):
            break

        # Space 키 (32): 화면 멈춤 / 재개 토글
        elif key == 32:
            is_frozen = not is_frozen
            status = "FROZEN (화면 멈춤)" if is_frozen else "RESUMED (실시간 재생)"
            print(f"[상태 변경] {status}")

        # F 키: FPS (프레임 수) 표시 토글
        elif key in (ord('f'), ord('F')):
            show_fps = not show_fps
            status = "ON (화면 표시)" if show_fps else "OFF (숨김)"
            print(f"[FPS 표시 토글] {status}")

        # 숫자키 1, 2, 3, 4: 특정 해상도로 변경 (1: 가장 낮음 -> 4: 가장 높음)
        elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
            target_idx = int(chr(key)) - 1
            res_name, target_w, target_h = RESOLUTIONS[target_idx]
            res_index = target_idx
            w, h, new_frame = set_camera_resolution(cap, detector, target_w, target_h, w, h)
            if new_frame is not None and not is_frozen:
                raw_frame = new_frame.copy()
                _, detected_faces = detector.detect(raw_frame)
            print(f"[화질 변경] {res_name} ({w}x{h}) 설정 완료")

        # R 키: 다음 화질로 순환 변경 (낮음 -> 높음 순으로 순환)
        elif key in (ord('r'), ord('R')):
            res_index = (res_index + 1) % len(RESOLUTIONS)
            res_name, target_w, target_h = RESOLUTIONS[res_index]
            w, h, new_frame = set_camera_resolution(cap, detector, target_w, target_h, w, h)
            if new_frame is not None and not is_frozen:
                raw_frame = new_frame.copy()
                _, detected_faces = detector.detect(raw_frame)
            print(f"[화질 변경] {res_name} ({w}x{h}) 설정 완료")

        # S 키: 현재 프레임 이미지로 저장
        elif key in (ord('s'), ord('S')):
            filename = f"capture_{w}x{h}_{'frozen' if is_frozen else 'live'}.png"
            cv2.imwrite(filename, display_frame)
            print(f"[저장 성공] 현재 화면이 '{filename}' 파일로 저장되었습니다.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()




