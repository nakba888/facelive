# 📝 연구일지 (1일차 상세 연구 노트)

**과제명**: OpenCV YuNet 딥러닝 기반 실시간 얼굴 감지 시스템 개발  
**작성일**: 2026년 08월 05일  
**연구자**: 연구개발팀  

---

## 🎯 1. 일일 연구 목표
- OpenCV `FaceDetectorYN` (YuNet) ONNX 모델을 활용한 실시간 얼굴 검출 모듈 수립
- 감지된 얼굴의 바운딩 박스(Bounding Box) 및 5개 랜드마크(양쪽 눈, 코끝, 입꼬리) 시각화 파이프라인 구축
- 얼굴 면적 기반 크기 순위 정렬 및 시각적 색상 표식 구현

---

## 🔬 2. 연구 및 실험 내용

### 2.1. YuNet ONNX 모델 조사 및 파이썬 연동
- 기존 Haar-Cascade 방식의 경우 조명 변화 및 고개의 각도(Pitch/Yaw/Roll) 변화에 취약하여 딥러닝 기반의 경량 모델인 **YuNet**을 선정함.
- `urllib.request` 모듈을 이용하여 모델 파일(`face_detection_yunet_2023mar.onnx`) 미존재 시 원격 저장소로부터 자동으로 다운로드하는 예외 처리 구문 작성.
- `cv2.FaceDetectorYN.create()` 메서드를 사용하여 탐지 임계값(`score_threshold=0.8`), NMS 임계값(`nms_threshold=0.3`)을 설정함.

### 2.2. 얼굴 면적(Area) 기반 무지개 범례(Rainbow Rank) 정렬 알고리즘 개발
- 탐지된 N개의 얼굴 결과 배열에서 너비와 높이($w \times h$)를 곱하여 카메라인 렌즈와 가깝고 화면을 많이 차지하는 얼굴순으로 내림차순 정렬(`valid_faces.sort`).
- 정렬된 1위 얼굴부터 무지개 색상 배열 `RAINBOW_COLORS` (빨·주·노·초·파·남·보)를 순환 매핑하여 `#1`, `#2` 라벨 및 바운딩 박스 시각화 구현.

```python
# 핵심 정렬 및 무지개 색상 매핑 구문
valid_faces.sort(key=lambda f: f[2] * f[3], reverse=True)
for rank, face in enumerate(valid_faces, start=1):
    box_color = RAINBOW_COLORS[(rank - 1) % len(RAINBOW_COLORS)]
    cv2.rectangle(image, (x, y), (x + w, y + h), box_color, 2)
```

### 2.3. 화면 멈춤(Freeze) 시 잔상 방지 레이어 분리 구조 구현
- 프레임을 정지할 때 기존 시각화 레이어가 덮어씌워져 잔상이 생기던 현상을 해결하기 위해, 원본 카메라인 `raw_frame`과 OSD 렌더링용 `display_frame`을 독립 분리함.

---

## 📊 3. 실험 결과 및 결론
- 720p HD 해상도 환경에서 평균 30 FPS 이상의 처리 속도를 유지하며, 5개 주요 랜드마크 포인트 및 얼굴 크기순 라벨링이 정상 동작함을 확인.
- 딥러닝 추론 결과가 실시간 프레임 버퍼와 성공적으로 동기화됨.
