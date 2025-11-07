# Fire-Detection-for-Firefighting-Robot

YOLACT 기반 실시간 화재 감지 시스템. 소방 로봇용 화재/연기 탐지 및 ROS2 통합.

시연 영상: https://youtu.be/uMIc3li09EU

![Demo](https://github.com/user-attachments/assets/f231a8a9-ed23-4c7b-b9f5-542b950abb50)

## 주요 기능

- **실시간 화재/연기 감지**: YOLACT 기반 인스턴스 세그멘테이션
- **다양한 입력 소스 지원**: RTSP 스트림, RealSense D435, 웹캠, MJPEG
- **ROS2 통합**: `fire_alert` 토픽으로 화재 상태 발행
- **실시간 시각화**: 바운딩 박스, 마스크, 점수 표시

## 환경 설정

### 1. Anaconda 환경 생성

```bash
conda env create -f environment.yml
conda activate cap
```

### 2. DCNv2 설치

```bash
cd external/DCNv2
python setup.py build develop
```

### 3. Cython NMS 컴파일

```bash
cd utils
python setup.py build_ext --inplace
```

### 4. 모델 가중치 다운로드

`weights/` 디렉토리에 학습된 모델 가중치를 배치:
- `yolact_resnet101_smoke_26_200000.pth`

## 사용법

### RTSP 스트림 (Tapo CCTV)

```bash
python fire_detect_cctv.py
```

`fire_detect_cctv.py`에서 RTSP URL 설정:
```python
username = "8536048"
password = "qpalzm0000"
ip_address = "192.168.200.130"
rtsp_url = f"rtsp://{username}:{password}@{ip_address}:554/stream1"
```

### RealSense D435 + 웹 스트림

1. RealSense 스트림 서버 실행:
```bash
python d435.py
```

2. 웹 스트림 기반 화재 감지:
```bash
python webstream.py
```

### 웹캠

```bash
python inference.py
```

## ROS2 통합

화재 감지 시 `fire_alert` 토픽으로 상태 발행:
- `1`: 화재 감지
- `0`: 정상

토픽 구독 예시:
```bash
ros2 topic echo /fire_alert
```

## 설정

### 모델 설정

`fire_detect_cctv.py`, `inference.py`, `webstream.py`에서 설정:

```python
CONFIG_NAME = 'yolact_resnet101_smoke_config'
WEIGHTS_PATH = 'weights/yolact_resnet101_smoke_26_200000.pth'
```

### 감지 클래스

현재 모델이 감지하는 클래스:
- Black smoke (검정색 연기)
- Gray smoke (회색 연기)
- White smoke (흰색 연기)
- Fire (화재)
- Cloud, Fog, Light, Sunlight 등

### 임계값 조정

```python
score_threshold=0.2  # 탐지 점수 임계값
```

## 프로젝트 구조

```
.
├── yolact.py              # YOLACT 모델 정의
├── fire_detect_cctv.py    # RTSP 스트림 화재 감지
├── inference.py           # 웹캠 기반 추론
├── webstream.py           # MJPEG 스트림 기반 감지
├── d435.py                # RealSense D435 스트림 서버
├── cctv.py                # RTSP 스트림 테스트
├── data/                  # 데이터셋 설정
├── layers/                # 네트워크 레이어
├── utils/                 # 유틸리티 함수
└── external/              # 외부 라이브러리 (DCNv2)
```

## 요구사항

- Python 3.8
- PyTorch >= 1.0.1
- CUDA 지원 GPU (권장)
- ROS2 (선택사항, ROS2 통합 사용 시)
- OpenCV
- RealSense SDK (D435 사용 시)

## 참고사항

- GPU 사용 시 CUDA 설정 확인 필요
- RTSP 스트림은 네트워크 상태에 따라 지연 가능
- RealSense D435 사용 시 `pyrealsense2` 설치 필요
