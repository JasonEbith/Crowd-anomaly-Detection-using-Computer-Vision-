import cv2
import numpy as np
import os
import requests  # <-- IMPORTANT: needed for Telegram HTTP requests
import time


# Telegram Configuration

BOT_TOKEN = "***********"
CHAT_ID = "***********"
API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

def send_telegram_alert(image_path, message):
    """Send Telegram alert with image and message"""
    try:
        with open(image_path, "rb") as photo:
            data = {"chat_id": CHAT_ID, "caption": message}
            response = requests.post(API_URL, data=data, files={"photo": photo})
        if response.status_code == 200:
            print(f" Alert sent to Telegram: {message}")
        else:
            print(f"[❌] Telegram Error: {response.text}")
    except Exception as e:
        print(f"[⚠️] Failed to send alert: {e}")


# Video & Processing Setup

VIDEO_PATH = "UNM1.mp4"
OUTPUT_DIR = "alerts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
ret, prev_frame = cap.read()
if not ret:
    raise SystemExit("Could not open video or read first frame.")

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
frame_idx = 0
alert_count = 0

# Detection sensitivity thresholds
MAG_THRESHOLD = 2.0
DIV_THRESHOLD = 2.0

# Cooldown to prevent Telegram flooding
last_alert_time = 0
ALERT_COOLDOWN = 10  # seconds between alerts

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Optical Flow Calculation
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    fx, fy = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy)

    # Divergence calculation
    dfx_dx = np.gradient(fx, axis=1)
    dfy_dy = np.gradient(fy, axis=0)
    divergence = dfx_dx + dfy_dy

    mean_mag = float(np.mean(mag))
    mean_div = float(np.mean(np.abs(divergence)))

    # Visualization
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    overlay = cv2.addWeighted(frame, 0.6, flow_rgb, 0.4, 0)

    abnormal = False
    if mean_mag > MAG_THRESHOLD or mean_div > DIV_THRESHOLD:
        abnormal = True
        alert_count += 1
        cv2.putText(overlay, "ABNORMAL CROWD MOTION DETECTED!", (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)

        image_path = os.path.join(OUTPUT_DIR, f"alert_{frame_idx:05d}.jpg")
        cv2.imwrite(image_path, overlay)
        print(f"[ALERT] Frame {frame_idx}: mean_mag={mean_mag:.2f}, mean_div={mean_div:.2f}")

        # Send Telegram alert if cooldown has passed
        current_time = time.time()
        if current_time - last_alert_time > ALERT_COOLDOWN:
            send_telegram_alert(image_path, f"🚨 Abnormal Crowd Motion Detected! ")
            last_alert_time = current_time

    # Debug text
    cv2.putText(overlay, f"mean_mag={mean_mag:.2f}, mean_div={mean_div:.2f}",
                (40, overlay.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Crowd Behavior Detection", overlay)
    prev_gray = gray

    if cv2.waitKey(1) & 0xFF == 27:
        break
    

cap.release()
cv2.destroyAllWindows()
print(f"Done. {alert_count} abnormal frames saved in '{OUTPUT_DIR}'")
