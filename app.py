import os
import cv2
import requests
from ultralytics import YOLO
from azure.eventhub import EventHubConsumerClient
from azure.storage.blob import BlobServiceClient

print("\n🚀 Starting container...")

# --- CONFIG ---
vehicle_classes = ['car', 'truck', 'motorbike']
distance_m = 20
line_y1 = 423
line_y2 = 555

# --- CONNECTION STRINGS ---
eventhub_conn = os.getenv("EVENT_HUB_CONNECTION", "")
storage_conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")

if not eventhub_conn:
    print("⚠️ EVENT_HUB_CONNECTION missing!")
if not storage_conn:
    print("⚠️ AZURE_STORAGE_CONNECTION_STRING missing!")

# --- YOLO SETUP ---
model = YOLO("yolov8n.pt")
print("✅ YOLOv8 model loaded on CPU")

# --- EVENT HANDLER ---
def on_event(partition_context, event):
    video_name = event.body_as_str()
    print(f"\n📩 Event received: {video_name}")

    # Download video
    url = f"https://highwayfootagestorage.blob.core.windows.net/processed-videos/{video_name}"
    local_video = f"/tmp/{video_name}"

    try:
        r = requests.get(url)
        r.raise_for_status()
        with open(local_video, "wb") as f:
            f.write(r.content)
        print(f"📥 Downloaded: {video_name} ({len(r.content)} bytes)")
    except Exception as e:
        print(f"❌ Failed to download {video_name}:", e)
        return

    # Analyze video
    cap = cv2.VideoCapture(local_video)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_name}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"🎞️ FPS: {fps}, Resolution: {w}x{h}")

    frame_number = 0
    total_vehicles = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % 10 == 0:
            detections = model(frame, verbose=False)[0]
            for r in detections.boxes:
                cls_id = int(r.cls.item())
                class_name = model.names[cls_id]
                if class_name in vehicle_classes:
                    total_vehicles += 1

        frame_number += 1

    cap.release()
    print(f"✅ Completed: {video_name} | Total vehicles detected: {total_vehicles}")

    # Create and upload log
    log_line = f"✅ Completed: {video_name} | Total vehicles detected: {total_vehicles}\n"
    log_filename = f"{video_name}.log"
    local_log = f"/tmp/{log_filename}"

    try:
        with open(local_log, "w") as f:
            f.write(log_line)

        blob_service = BlobServiceClient.from_connection_string(storage_conn)
        blob_client = blob_service.get_blob_client(container="test-logs", blob=log_filename)

        with open(local_log, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        print(f"📝 Log uploaded to 'test-logs/{log_filename}'")

    except Exception as e:
        print("⚠️ Failed to upload log:", e)

# --- EVENT HUB LISTENER ---
try:
    client = EventHubConsumerClient.from_connection_string(
        conn_str=eventhub_conn,
        consumer_group="$Default",
        eventhub_name="video-segments"
    )
    with client:
        client.receive(on_event=on_event, starting_position="-1")

except Exception as e:
    print("❌ Failed to connect to Event Hub:", e)

print("✅ Container started successfully")
