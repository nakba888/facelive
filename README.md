# FaceLive 📸

OpenCV **YuNet ONNX 딥러닝 모델**을 활용한 인터랙티브 실시간 웹캠 얼굴 감지 애플리케이션입니다.

---

## 🌟 주요 기능

- 🤖 **딥러닝 얼굴 감지 & 5-Landmarks**: 눈, 코끝, 양쪽 입꼬리 좌표 시각화
- 🌈 **얼굴 크기별 무지개 순위 표식**: 화면에 가까운 큰 얼굴부터 `#1`, `#2`, `#3` 순으로 무지개 색상(**빨·주·노·초·파·남·보**) 및 라벨 시각화
- ⏸️ **잔상 없는 화면 정지 (Freeze/Pause)**: `Space Bar`를 누르면 중복 텍스트 겹침 없이 깔끔하게 화면 멈춤/재개
- 📱 **가로 / 세로 화면 비율 모드 (`V` 키)**: 
  - 가로 16:9 기본
  - 세로 90° 시계방향 회전
  - 세로 90° 반시계방향 회전
  - 세로 9:16 중앙 크롭
- 🔌 **내장 웹캠 / USB 외장 웹캠 지원 (`C` 키)**: `python app.py 1` 또는 실행 중 `C` 키로 카메라 즉시 전환
- ⚙️ **실시간 해상도/화질 조절 (`1` ~ `4`, `R` 키)**: `1` (240p) ~ `4` (1080p FHD) 화질 즉시 변경
- ⚡ **실시간 FPS 표시/숨김 (`F` 키)**: 프레임 수 측정 배지 ON/OFF
- 📸 **날짜/시간 타임스탬프 이미지 저장 (`S` 키)**: `captures/` 전용 폴더에 `capture_년월일_시분초_...png` 파일명으로 보존

---

## 📂 프로젝트 구조

```text
facelive/
├── app.py                         # 메인 실행 스크립트
├── requirements.txt               # 의존성 라이브러리 목록
├── face_detection_yunet_2023mar.onnx # YuNet 딥러닝 모델 (자동 다운로드)
├── captures/                      # 캡처 이미지 저장 폴더 (S키)
├── .gitignore                     # Git 제외 설정 파일
└── README.md                      # 프로젝트 안내 문서
```

---

## 🛠 설치 가이드

### [방법 A] Miniconda / Anaconda 사용 시 (추천)

```bash
# 1. 저장소 클론
git clone https://github.com/nakba888/facelive.git
cd facelive

# 2. 가상 환경 생성 및 활성화
conda create -n facelive python=3.12 -y
conda activate facelive

# 3. 필요 패키지 설치
pip install -r requirements.txt
```

### [방법 B] Python 기본 venv 사용 시

```bash
git clone https://github.com/nakba888/facelive.git
cd facelive

python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows 기준

pip install -r requirements.txt
```

---

## 🚀 실행 방법

### 기본 실행 (내장 웹캠)
```bash
python app.py
```

### USB 외장 웹캠 지정 실행
```bash
python app.py 1
```

> **참고**: 실행 시 YuNet 모델 파일(`face_detection_yunet_2023mar.onnx`)이 자동 다운로드됩니다.

---

## ⌨️ 단축키 요약 표

| 단축키 | 기능 설명 |
| :--- | :--- |
| **`Space Bar`** | 화면 멈춤(Freeze) / 다시 재생(Resume) |
| **`V`** | 가로 ↔ 세로 화면 비율 변경 (가로, 90° 시계/반시계 회전, 9:16 크롭) |
| **`1` ~ `4`** | 화질 변경 (`1`: 240p, `2`: 480p, `3`: 720p HD, `4`: 1080p FHD) |
| **`R`** | 화질(해상도) 순환 변경 |
| **`C`** | 카메라 전환 (0: 내장웹캠 ↔ 1: USB웹캠) |
| **`F`** | 실시간 FPS(프레임 수) 표시 ON / OFF |
| **`S`** | 현재 화면을 `captures/` 폴더에 날짜/시간 파일명으로 저장 |
| **`Q` / `ESC`** | 프로그램 종료 |
