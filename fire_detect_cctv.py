import cv2
import numpy as np
import torch
from yolact import Yolact
from utils.augmentations import FastBaseTransform
from data import cfg, set_cfg
from eval import prep_display
import eval
from argparse import Namespace
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

# ROS2 fire alert publisher node
class FirePublisher(Node):
    def __init__(self):
        super().__init__('fire_detector_node')
        self.publisher_ = self.create_publisher(Int32, 'fire_alert', 10)

    def publish_fire_status(self, is_fire):
        msg = Int32()
        msg.data = 1 if is_fire else 0
        self.publisher_.publish(msg)

# YOLACT 설정
eval.args = Namespace(
    top_k=5,
    display_lincomb=False,
    crop=True,
    score_threshold=0.2,
    display_fps=True,
    display_masks=True,
    display_text=True,
    display_bboxes=True,
    display_scores=True
)

CONFIG_NAME = 'yolact_resnet101_smoke_config'
WEIGHTS_PATH = 'weights/yolact_resnet101_smoke_26_200000.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 로드
set_cfg(CONFIG_NAME)
cfg.mask_proto_debug = False
net = Yolact()
net.load_weights(WEIGHTS_PATH)
net.eval()
net = net.to(DEVICE)

transform = FastBaseTransform()

# ROS2 초기화
rclpy.init()
fire_node = FirePublisher()

# Tapo 카메라 설정
username = "8536048"
password = "qpalzm0000"
ip_address = "192.168.200.130"
rtsp_url = f"rtsp://{username}:{password}@{ip_address}:554/stream1"

# 스트리밍 시작
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("❌ 카메라 연결 실패! RTSP 주소를 확인하세요.")
    exit()

print("✅ Tapo 스트리밍 시작!")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 프레임 수신 실패!")
            continue

        # YOLACT 모델 입력 준비
        frame_tensor = torch.from_numpy(frame).cuda().float()
        batch = transform(frame_tensor.unsqueeze(0))

        with torch.no_grad():
            preds = net(batch)

        result = prep_display(preds, frame_tensor, None, None, undo_transform=False)

        class_names = cfg.dataset.class_names
        target_keywords = ['fire', 'smoke']

        is_fire = False

        if preds is not None and preds[0] is not None:
            detection = preds[0].get('detection', None)

            if detection is not None and 'class' in detection and 'score' in detection:
                classes = detection['class'].cpu().numpy()
                scores = detection['score'].cpu().numpy()

                for cls_id, score in zip(classes, scores):
                    cls_name = class_names[cls_id] if cls_id < len(class_names) else 'unknown'

                    if any(keyword in cls_name.lower() for keyword in target_keywords) and score >= 0.5:
                        is_fire = True
                        print(f"🔥 화재 탐지: {cls_name} (score={score:.2f})")
                        break
            # else:
            #     print("👀 탐지 결과 없음 or detection=None")

        # ROS2 토픽 발행
        fire_node.publish_fire_status(is_fire)

        cv2.imshow("Fire Detection", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    fire_node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows() 