import cv2

# --- 사용자 설정 ---
username = "8536048"            # Tapo 앱에서 만든 계정
password = "qpalzm0000"          # Tapo 앱에서 만든 비밀번호
ip_address = "192.168.200.130"   # Tapo 카메라의 IP 주소

# RTSP 스트림 주소
rtsp_url = f"rtsp://{username}:{password}@{ip_address}:554/stream1"

# 스트리밍 시작
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("❌ 카메라 연결 실패! RTSP 주소를 확인하세요.")
    exit()

print("✅ Tapo 스트리밍 시작!")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ 프레임 수신 실패!")
        break

    # 화면에 표시
    cv2.imshow("Tapo C210 Live", frame)

    # 'q' 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료 처리
cap.release()
cv2.destroyAllWindows()