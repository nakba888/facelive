import os
import sys
import time
from datetime import datetime
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

def open_camera(cam_id):
    """지정한 카메라 ID(0: 내장웹캠, 1: USB웹캠)를 안전하게 연결하고 자동 초점을 활성화합니다."""
    cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_id)

    if cap.isOpened():
        # 카메라 자동 초점(Auto-Focus) 활성화 시도
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    return cap


# 화면 방향 (가로 / 세로 회전 / 세로 크롭) 모드 정의
ORIENTATIONS = [
    ("가로 (16:9 기본)", 0),
    ("세로 (90° 시계방향 회전)", 1),
    ("세로 (90° 반시계방향 회전)", 2),
    ("세로 (9:16 중앙 크롭)", 3),
]


def transform_frame(frame, mode):
    """선택한 방향 모드에 따라 프레임을 회전하거나 세로 9:16 비율로 크롭합니다."""
    if frame is None:
        return None

    if mode == 1:
        # 90도 시계방향 회전 (카메라를 옆으로 세워 고정했을 때)
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif mode == 2:
        # 90도 반시계방향 회전
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif mode == 3:
        # 가로 화면의 중앙을 세로 9:16 비율로 자르기 (카메라는 그대로 두고 세로 뷰 포커스)
        h, w, _ = frame.shape
        target_w = int(h * (9.0 / 16.0))
        if target_w < w:
            start_x = (w - target_w) // 2
            return frame[:, start_x:start_x + target_w]
        return frame

    return frame


def main():
    # 1. YuNet 모델 다운로드 검사
    download_model_if_needed()

    # CLI 인자로 카메라 번호 지정 가능 (예: python app.py 1)
    cam_id = 0
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        cam_id = int(sys.argv[1])

    # 2. 웹캠 연결 (Windows DSHOW 백엔드 우선 적용)
    cap = open_camera(cam_id)

    # 지정한 카메라 연결 실패 시 다른 번호로 백업 연결 시도 (1번 시도 후 0번)
    if not cap.isOpened():
        fallback_id = 0 if cam_id != 0 else 1
        print(f"[알림] 카메라 {cam_id}번 연결 실패. 카메라 {fallback_id}번으로 연결 시도합니다...")
        cap = open_camera(fallback_id)
        if cap.isOpened():
            cam_id = fallback_id

    if not cap.isOpened():
        print("[Error] 카메라는 연결할 수 없습니다. USB 웹캠 연결을 확인해 주세요.")
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

    is_frozen = False
    res_index = 2       # 기본값: 3번 720p HD
    orient_index = 0    # 기본값: 가로 (16:9 기본)
    show_fps = True     # F 키로 FPS 표시 토글
    prev_time = time.time()
    fps = 0.0

    # 초기 방향 변환 적용
    raw_frame = transform_frame(frame, orient_index)
    h, w, _ = raw_frame.shape

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

    detected_faces = None
    face_count = 0

    WINDOW_NAME = "YuNet Face Detection - facelive"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 540)  # 노트북 화면에 잘 맞도록 초기 창 크기 설정

    print("\n" + "=" * 65)
    print("      YuNet OpenCV 얼굴 감지 & 가로/세로 비율 조절 (facelive)")
    print("=" * 65)
    print(" 조작 방법:")
    print("   [Space Bar]   : 화면 정지(Pause) / 다시 재생(Resume)")
    print("   [V]           : 가로/세로 화면 비율 변경")
    print("                     (가로 -> 세로90°시계 -> 세로90°반시계 -> 세로9:16크롭)")
    print("   [1 / 2 / 3 / 4]: 화질(해상도) 즉시 변경 (1:낮음 -> 4:높음)")
    print("   [R]           : 다음 화질로 순환 변경")
    print("   [C]           : 카메라 전환 (0: 내장웹캠 <-> 1: USB웹캠)")
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

            # 세로/가로 방향 변환 적용
            transformed = transform_frame(frame, orient_index)
            if transformed is None:
                continue

            cur_h, cur_w, _ = transformed.shape
            # 변환 후 프레임 크기가 변경된 경우 입력 크기 업데이트
            if (cur_w, cur_h) != (w, h):
                w, h = cur_w, cur_h
                detector.setInputSize((w, h))

            raw_frame = transformed.copy()
            # YuNet 얼굴 감지 수행 (변환된 세로/가로 프레임에서 감지)
            _, detected_faces = detector.detect(raw_frame)

        # -------------------------------------------------------------
        # 매 루프마다 raw_frame에서 새로 복사하여 시각화 (텍스트 겹침 방지)
        # -------------------------------------------------------------
        display_frame = raw_frame.copy()

        # 1. 얼굴 및 순위(#1, #2...) 그리기
        face_count = draw_faces(display_frame, detected_faces, score_threshold=0.8)

        # 2. 상단 좌측: 상태 및 카메라 번호 표시
        cam_type_str = "USB Cam #1" if cam_id == 1 else f"Cam #{cam_id}"
        if not is_frozen:
            cv2.circle(display_frame, (25, 25), 8, (0, 255, 0), -1)
            cv2.putText(
                display_frame, f"LIVE ({cam_type_str}) | Faces: {face_count}", (42, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA
            )
        else:
            cv2.circle(display_frame, (25, 25), 8, (0, 0, 255), -1)
            cv2.putText(
                display_frame, f"FROZEN ({cam_type_str}) | Faces: {face_count}", (42, 31),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA
            )

        # 3. 상단 우측: 현재 해상도 & 방향 표시
        res_name = RESOLUTIONS[res_index][0]
        orient_name = ORIENTATIONS[orient_index][0].split(" ")[0]  # 예: "세로" 또는 "가로"
        res_info = f"{w}x{h} ({orient_name} | {res_name})"
        (res_w, res_h), _ = cv2.getTextSize(res_info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(
            display_frame,
            (w - res_w - 20, 10),
            (w - 10, 40),
            (40, 40, 40), -1
        )
        cv2.putText(
            display_frame, res_info, (w - res_w - 15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA
        )

        # 4. 상단 우측: FPS 표시 배지 (F키로 토글 가능)
        if show_fps:
            fps_info = f"FPS: {fps:.1f}"
            (fps_w, _), _ = cv2.getTextSize(fps_info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            fps_x = w - res_w - fps_w - 45
            cv2.rectangle(
                display_frame,
                (fps_x - 10, 10),
                (fps_x + fps_w + 10, 40),
                (40, 40, 40), -1
            )
            cv2.putText(
                display_frame, fps_info, (fps_x, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA
            )

        # 5. 하단 조작 가이드 안내
        guide_text = "[Space]: Freeze  |  [V]: Vert/Horiz  |  [1-4]: Res  |  [C]: Cam  |  [F]: FPS"
        cv2.putText(
            display_frame, guide_text, (15, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
        )

        # 화면 출력
        cv2.imshow(WINDOW_NAME, display_frame)

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

        # V 키: 가로/세로 화면 비율 모드 변경
        elif key in (ord('v'), ord('V')):
            orient_index = (orient_index + 1) % len(ORIENTATIONS)
            orient_name = ORIENTATIONS[orient_index][0]
            print(f"[화면 방향 변경] {orient_name} 적용")

            # 새로운 방향으로 프레임 크기 업데이트 및 감지기 재설정
            if not is_frozen and raw_frame is not None:
                transformed = transform_frame(frame, orient_index)
                if transformed is not None:
                    w, h = transformed.shape[1], transformed.shape[0]
                    detector.setInputSize((w, h))
                    raw_frame = transformed.copy()
                    _, detected_faces = detector.detect(raw_frame)

                    # 세로 모드일 경우 노트북 화면에 잘 맞도록 창 크기 조정
                    if orient_index != 0:
                        cv2.resizeWindow(WINDOW_NAME, 450, 800)
                    else:
                        cv2.resizeWindow(WINDOW_NAME, 960, 540)

        # C 키: 카메라 0번(내장) <-> 1번(USB) 실시간 전환
        elif key in (ord('c'), ord('C')):
            next_cam_id = 1 if cam_id == 0 else 0
            print(f"[카메라 전환] 카메라 {next_cam_id}번으로 전환 시도 중...")
            cap.release()
            new_cap = open_camera(next_cam_id)
            if new_cap.isOpened():
                target_w, target_h = RESOLUTIONS[res_index][1], RESOLUTIONS[res_index][2]
                cam_id = next_cam_id
                cap = new_cap
                w, h, new_frame = set_camera_resolution(cap, detector, target_w, target_h, w, h)
                if new_frame is not None and not is_frozen:
                    transformed = transform_frame(new_frame, orient_index)
                    if transformed is not None:
                        w, h = transformed.shape[1], transformed.shape[0]
                        detector.setInputSize((w, h))
                        raw_frame = transformed.copy()
                        _, detected_faces = detector.detect(raw_frame)
                print(f"[카메라 전환 완료] 카메라 #{cam_id} 연결 성공!")
            else:
                print(f"[경고] 카메라 #{next_cam_id}번을 연결할 수 없어 기존 카메라 #{cam_id}를 유지합니다.")
                cap = open_camera(cam_id)

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
                transformed = transform_frame(new_frame, orient_index)
                if transformed is not None:
                    w, h = transformed.shape[1], transformed.shape[0]
                    detector.setInputSize((w, h))
                    raw_frame = transformed.copy()
                    _, detected_faces = detector.detect(raw_frame)
            print(f"[화질 변경] {res_name} ({w}x{h}) 설정 완료")

        # R 키: 다음 화질로 순환 변경
        elif key in (ord('r'), ord('R')):
            res_index = (res_index + 1) % len(RESOLUTIONS)
            res_name, target_w, target_h = RESOLUTIONS[res_index]
            w, h, new_frame = set_camera_resolution(cap, detector, target_w, target_h, w, h)
            if new_frame is not None and not is_frozen:
                transformed = transform_frame(new_frame, orient_index)
                if transformed is not None:
                    w, h = transformed.shape[1], transformed.shape[0]
                    detector.setInputSize((w, h))
                    raw_frame = transformed.copy()
                    _, detected_faces = detector.detect(raw_frame)
            print(f"[화질 변경] {res_name} ({w}x{h}) 설정 완료")

        # S 키: 현재 프레임 이미지로 저장 (captures/ 폴더에 날짜 및 시간 타임스탬프 파일명 적용)
        elif key in (ord('s'), ord('S')):
            os.makedirs("captures", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}_{w}x{h}_{'frozen' if is_frozen else 'live'}.png"
            filepath = os.path.join("captures", filename)
            cv2.imwrite(filepath, display_frame)
            print(f"[저장 성공] 현재 화면이 '{filepath}' 파일로 저장되었습니다.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()






