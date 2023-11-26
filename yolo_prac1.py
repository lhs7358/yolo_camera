import cv2
import torch

# YOLO 모델 로드 (사전 훈련된 모델 사용)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 카메라 설정
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 모델을 사용하여 프레임에서 객체 감지
    results = model(frame)

    # 결과에서 'person' 클래스 필터링
    results = results.xyxy[0].numpy()  # 감지된 객체들의 바운딩 박스
    for result in results:
        if result[5] == 0:  # 클래스 ID '0'이 'person'임 (YOLO COCO 데이터셋 기준)
            x1, y1, x2, y2 = map(int, result[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 결과 보여주기
    cv2.imshow('YOLO People Detection', frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
