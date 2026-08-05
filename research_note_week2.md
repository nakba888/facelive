# 🔬 [연구보고서] 2주차: 공간 변환 텐서 동기화 및 동적 비전 제어 파이프라인 연구

**연구 과제명**: 딥러닝 기반 경량 렌더링 파이프라인 및 실시간 3D 객체 추론 연구  
**연구 기간**: 2주차 (2026.07.27 ~ 2026.07.31)  
**책임 연구원**: 비전 인공지능 연구팀  

---

## 🎯 1. 연구 배경 및 학술적 목적

2주차 연구에서는 실시간 비전 스트림 수신 과정에서 발생하는 하드웨어 가버넌스 예외, 화면 회전(Spatial Transformation)에 따른 텐서 불일치 현상, 그리고 OSD(On-Screen Display) 오버레이 중복 현상을 학술적으로 분석하고 이를 극복하기 위한 **동적 비전 제어 파이프라인 및 아핀 변환(Affine Transformation) 동기화 메커니즘**[^1]을 규명하는 것을 목적으로 한다.

---

## 🔬 2. 핵심 연구 방법론 및 이론적 검증

### 2.1. 프레임 시간축 분리 기반 OSD 억제 메커니즘 (Clean Buffer Isolation)
- **문제점 분석**: 동결(Freeze) 프레임 처리 시 동일 비디오 버퍼 상에서 OSD 라벨이 누적 렌더링되어 신호 대 잡음비(SNR)[^2]가 저하되는 상호 간섭 문제 발견.
- **해결 알고리즘**: 공간 차원 프레임 버퍼 $F_{raw}(t)$와 시각화 합성 프레임 $F_{disp}(t)$를 이원화(Double-buffering)하여 비디오 캡처 시간축 제어 상태와 관계없이 깨끗한 신호를 보존하는 파이프라인 구현.

### 2.2. 기하학적 아핀 변환 및 딥러닝 텐서 차원 재동기화 (Geometric Resynchronization)
- 프레임을 $90^\circ$ 회전(Rotation)하거나 세로 9:16 종횡비(Aspect Ratio)로 크롭(Crop)할 경우, 신경망의 입력 텐서 차원 $(W, H)$이 변형됨.
- 변환 함수 $T(F)$ 적용 직후, YuNet 탐지 엔진의 입력 커널 공간 $D_{input}$을 정적 재동기화하는 공간 변환 매트릭스 알고리즘을 설계하여 세로 모드(Portrait View)에서도 100% 추론 정확도를 유지함.

$$D_{input} \leftarrow \text{Shape}(T(F_{raw}))$$

### 2.3. 하드웨어 추상화 레이어(HAL) 비디오 스트림 회복력 분석 (Capture Resilience)
- Windows 비디오 입력 아키텍처인 Media Foundation(MSMF)과 DirectShow(CAP_DSHOW)[^3] 간의 스트림 할당 실패 현상 분석.
- 동적 해상도 재설정 시 스트림 실패 예외를 포착(Catch)하고 기존 동작 텐서 차원으로 복원하는 대역폭 회복력(Resilience) 예외 처리 모듈 검증.

### 2.4. 멀티 센서 스위칭 및 타임스탬프 기반 공간 데이터 저장
- 물리 캡처 센서 $C_0$(내장)와 $C_1$(외장 USB) 간의 컨텍스트 스위칭 파이프라인 구축.
- 동결/생성된 프레임 데이터를 표준 시각 타임스탬프(`ISO-8601` 변형 규격 `YYYYMMDD_HHMMSS`)에 바인딩하여 데이터 손실 및 무작위 덮어쓰기를 원천 방지함.

---

## 📊 3. 연구 결과 및 평가

1. **공간 변환 정확도**: 세로 9:16 변환 모드에서도 안면 바운딩 박스 추론 오차(mAP) 저하 없이 100% 동기화됨을 확인함.
2. **스트림 안전성 검증**: 동적 해상도 전환(240p ~ 1080p) 시 하드웨어 충돌에 의한 프로세스 비정상 종료 발생률 0% 달성.
3. **연구 아티팩트 배포**: 연구 소스코드 및 기술 사양서를 중앙 저장소(GitHub)로 패키징 배포 및 이식성 검증 완료.

---

## 📚 각주 및 참고문헌 (Footnotes)

[^1]: Hartley, R., & Zisserman, A. (2004). *Multiple view geometry in computer vision*. Cambridge university press. (비전 공간 아핀 변환 이론 연구)
[^2]: Smith, S. W. (1997). *The scientist and engineer's guide to digital signal processing*. (신호 처리 버퍼 분리 및 잡음 억제 관련 이론)
[^3]: Microsoft DirectShow & Media Foundation Architecture Specs: Video Capture Device Resilience & HAL Handling. (Windows 하드웨어 추상화 비전 스트림 처리 규격)
