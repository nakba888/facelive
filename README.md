# FaceLive 📸

OpenCV **YuNet ONNX 모델**을 활용한 실시간 웹캠 얼굴 감지 및 인터랙티브 제어 애플리케이션입니다.

---

## 🌟 주요 기능
- **실시간 얼굴 감지 & 5-Landmarks**: 눈, 코끝, 입꼬리 좌표 및 바운딩 박스 시각화
- **얼굴 크기별 무지개 순위 표식**: 화면에 가까운 큰 얼굴부터 `#1`, `#2`, `#3` 순으로 무지개 색상(빨·주·노·초·파·남·보) 표시
- **화면 정지(Pause/Freeze)**: `Space Bar`를 눌러 실시간 화면을 멈추고 캡처 상태로 분석
- **실시간 해상도/화질 즉시 변경**: `1` (240p) ~ `4` (1080p FHD) 키 또는 `R` 키로 웹캠 해상도 변경
- **실시간 FPS 표시/숨김**: `F` 키로 프레임 수(FPS) 배지 ON/OFF
- **화면 저장**: `S` 키를 눌러 현재 화면을 `.png` 이미지로 저장

---

## 🛠 설치 방법
```bash
git clone https://github.com/YOUR_USERNAME/facelive.git
cd facelive
pip install -r requirements.txt
```

---

## 🚀 실행 방법
```bash
python app.py
```
> 실행 시 필요한 YuNet 모델 파일(`face_detection_yunet_2023mar.onnx`)이 자동 다운로드됩니다.

---

## ⌨️ 단축키 안내
| 단축키 | 기능 설명 |
| :--- | :--- |
| **`Space Bar`** | 화면 멈춤(Freeze) / 다시 재생(Resume) |
| **`1` / `2` / `3` / `4`** | 화질(해상도) 즉시 변경 (1: 240p, 2: 480p, 3: 720p HD, 4: 1080p FHD) |
| **`R`** | 화질 순환 변경 |
| **`F`** | 실시간 FPS(프레임 수) 표시 ON / OFF |
| **`S`** | 현재 화면 이미지 파일로 저장 |
| **`Q` / `ESC`** | 프로그램 종료 |
