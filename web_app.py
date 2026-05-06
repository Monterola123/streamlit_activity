import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2
from collections import deque

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🎥 Smart Object Zone Detection System")
st.write("Detects objects and counts them by zone (Left / Center / Right)")


# Store detections
detections = deque(maxlen=20)


def get_zone(x_center, width):
    if x_center < width / 3:
        return "LEFT"
    elif x_center < (2 * width) / 3:
        return "CENTER"
    else:
        return "RIGHT"


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    height, width, _ = img.shape

    results = model.predict(img, conf=0.5, verbose=False)
    result = results[0]

    # 🔥 Zone counters
    zone_count = {"LEFT": 0, "CENTER": 0, "RIGHT": 0}

    # Draw zone lines
    cv2.line(img, (width // 3, 0), (width // 3, height), (255, 255, 255), 2)
    cv2.line(img, (2 * width // 3, 0), (2 * width // 3, height), (255, 255, 255), 2)

    if result.boxes is not None:
        boxes = result.boxes
        names = model.names

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            cls_id = int(box.cls[0])

            object_name = names[cls_id]
            x_center = (x1 + x2) / 2

            zone = get_zone(x_center, width)

            # 🔥 COUNT PER ZONE
            zone_count[zone] += 1

            label = f"{object_name} - {zone}"
            detections.append(label)

            # Draw bounding box
            cv2.rectangle(img,
                          (int(x1), int(y1)),
                          (int(x2), int(y2)),
                          (0, 255, 0), 2)

            cv2.putText(
                img,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

    # 🔥 DISPLAY ZONE COUNTS ON VIDEO
    cv2.putText(img, f"LEFT: {zone_count['LEFT']}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(img, f"CENTER: {zone_count['CENTER']}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.putText(img, f"RIGHT: {zone_count['RIGHT']}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")


# WebRTC Stream
webrtc_streamer(
    key="zone-counting-system",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)


# 📊 Bottom Display
