import cv2

cap = cv2.VideoCapture()
cap.open('rtsp://admin:jse17jse17@192.168.0.217:554/cam/realmonitor?channel=1&subtype=0')

while (True):
    ret, frame = cap.read()
    cv2.imshow("camCapture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break