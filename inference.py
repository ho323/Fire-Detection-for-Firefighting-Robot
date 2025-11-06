import cv2
import torch
from yolact import Yolact
from utils.augmentations import FastBaseTransform
from data import cfg, set_cfg
from layers.output_utils import postprocess
import numpy as np

# 1. 설정 --------------------------------------------------
CONFIG_NAME = 'yolact_resnet101_smoke_config'
WEIGHTS_PATH = 'weights/yolact_resnet101_smoke_26_200000.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 모델 로딩 ---------------------------------------------
set_cfg(CONFIG_NAME)
cfg.mask_proto_debug = False
cfg.eval_mask_branch = True

model = Yolact()
model.load_weights(WEIGHTS_PATH)
model.eval()
model = model.to(DEVICE)

transform = FastBaseTransform()

# 3. 결과 시각화 함수 ---------------------------------------
COLORS = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
          for _ in range(cfg.num_classes)]

def draw_detections(img, classes, scores, boxes, masks, threshold=0.3):
    h, w, _ = img.shape
    num_dets = classes.size(0)
    for i in range(num_dets):
        if scores[i] < threshold:
            continue

        x1, y1, x2, y2 = boxes[i, :].int()
        color = COLORS[int(classes[i]) % len(COLORS)]
        label = cfg.dataset.class_names[int(classes[i])]
        score = scores[i].item()

        # draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # draw text
        text = f"{label}: {score:.2f}"
        cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # draw mask
        if masks is not None:
            mask = masks[i].cpu().numpy()
            colored_mask = np.stack([mask * c for c in color], axis=2).astype(np.uint8)
            img = cv2.addWeighted(img, 1.0, colored_mask, 0.5, 0)
    return img

# 4. CCTV 스트림 열기 ----------------------------------------
cap = cv2.VideoCapture(0)  # 0번 webcam, 또는 rtsp/http 주소 입력 가능

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    with torch.no_grad():
        batch = transform(frame).to(DEVICE)
        preds = model(batch)
        h, w, _ = frame.shape
        classes, scores, boxes, masks = postprocess(preds, w, h, score_threshold=0.3)

    frame = draw_detections(frame, classes, scores, boxes, masks)
    cv2.imshow('YOLACT CCTV Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
