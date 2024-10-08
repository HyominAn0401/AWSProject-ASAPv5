# ASAPv5 
🎟️ 공연장 입장 시 위해물품 탐지 서비스  

## 간단 소개
X-Ray 영상 속 위해 물품을 YOLO 모델로 탐지하고 알람을 송출하는 서비스

## 프로젝트 개요
- **기간**: 2024.04.27 ~ 2024.05.25
- **개발 인원**: 5명
- **개발 언어**: Python 3.12
- **개발 환경**: AWS SageMaker, Lambda, API Gateway, Cloud9
- **데이터베이스**: S3
- **모델**: YOLOv8
- **인증 및 권한**: AWS IAM
- **프론트엔드**: Streamlit

---

## 서비스 요구사항

### 1. HOME
- 서비스 메뉴 설명을 출력

### 2. SETTING
- X-Ray 영상을 업로드하여 S3 버킷에 저장
- 지원되는 파일 형식은 **mp4, mpeg4, avi**

### 3. DETECTION
- 업로드한 영상이 재생됨
- YOLOv8 모델이 위해물품을 탐지한 경우, 알람을 출력

### 4. DASHBOARD
- 탐지된 위해물품의 결과를 대시보드로 시각화하여 출력

---

## 트러블슈팅

### 1. Presigned URL
- **문제**: 10MB 이상의 영상을 업로드할 때 AWS API Gateway의 기본 요청 크기 제한을 초과하는 문제가 발생
- **해결**: Presigned URL을 사용하여 서버를 거치지 않고 파일을 S3에 직접 업로드
- **Presigned URL의 유효기간**: 3600초(1시간)로 설정한 이유는 파일 업로드가 예상 시간 내에 완료되도록 하기 위함

### 2. ~~Import CV2~~
- ~~문제 해결 방법: 삭제 후 다시 실행하면 됨. 참고: 999 채널의 답변~~

### 3. ~~Timeout~~
- ~~문제 해결 방법: SageMaker Studio에서 실행~~

---

## 모델 설명

### 모델 아키텍처: YOLOv8
- **YOLOv8을 선택한 이유**:
  1. YOLOv5, YOLOv7, YOLOv8을 비교한 결과, **Recall 값**과 **mAP**를 기준으로 YOLOv8s 모델을 선택
  2. 위해물품을 탐지하는 경우, **위해물품을 놓치는 것**이 위해물품이 아닌 것을 잘못 탐지하는 것보다 더 큰 문제라고 판단함

### 훈련 데이터
- **X-Ray 이미지 데이터셋**을 사용하여 YOLOv8 모델을 훈련

---

## 담당한 부분

1. **영상 업로드 및 출력** 기능 구현
2. **SageMaker 엔드포인트 생성** 및 모델 배포
3. **아키텍처 설계** 및 구현

---
