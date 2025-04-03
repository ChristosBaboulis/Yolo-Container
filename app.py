print("🚀 script started", flush=True)
import asyncio
from azure.eventhub.aio import EventHubConsumerClient
import requests
import os
from azure.storage.blob import BlobServiceClient
import datetime
from azure.eventhub.extensions.checkpointstoreblobaio import BlobCheckpointStore
import cv2
from ultralytics import YOLO
import time
import json
print("🔧 imports done", flush=True)

vehicle_classes = ['car', 'truck', 'motorbike']
distance_m = 20
line_y1 = 423
line_y2 = 555


conn_str = os.getenv("EVENT_HUB_CONNECTION", "")
storage_conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
checkpoint_store = BlobCheckpointStore.from_connection_string(
    storage_conn, "eventhub-checkpoints"
)

print("🔧 conn_str done", flush=True)
print(f"🔑 conn_str = {conn_str[:30]}...", flush=True)

async def on_event(partition_context, event):
    try:
        print(f"📩 Received on partition {partition_context.partition_id}", flush=True)
        video_name_raw = event.body_as_str()
        print(f"📦 Message Raw: {video_name_raw}", flush=True)

        # Default: use raw string (σε περίπτωση που δεν είναι JSON)
        video_name = video_name_raw

        # Try to parse JSON-style event from Event Grid
        try:
            parsed = json.loads(video_name_raw)
            if isinstance(parsed, list) and "subject" in parsed[0]:
                subject = parsed[0]["subject"]
                video_name = subject.split("/")[-1]
                print(f"🧠 Extracted from JSON subject: {video_name}", flush=True)
        except Exception as e:
            print(f"⚠️ Not JSON format or parsing failed: {e}", flush=True)


        # --- DOWNLOAD VIDEO ---
        local_video = f"/tmp/{video_name}"

        try:
            blob_service = BlobServiceClient.from_connection_string(storage_conn)
            blob_client = blob_service.get_blob_client(container="processed-videos", blob=video_name)

            with open(local_video, "wb") as f:
                download_stream = blob_client.download_blob()
                f.write(download_stream.readall())

            print(f"📥 Downloaded video via SDK: {video_name}", flush=True)

        except Exception as e:
            print(f"❌ Failed to download {video_name} via SDK: {e}", flush=True)
            return

        cap = cv2.VideoCapture(local_video)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_name}")
            return

        model = YOLO("yolov8n.pt")
        tracked = {}
        firstLineFrame = {}
        speed_log = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(frame, persist=True, verbose=False)[0]
            
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            if frame_number < 25:
                continue

            results = model.track(frame, persist=True, verbose=False)[0]

            for box in results.boxes:
                cls = int(box.cls[0].item())
                name = model.names[cls]
                if name not in vehicle_classes:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                direction = "bottom_to_top" if cx < frame_width // 2 else "top_to_bottom"

                track_id = int(box.id[0].item())

                # Αγνοούμε αν το κέντρο είναι έξω από την περιοχή
                if direction == "bottom_to_top":
                    if cy < line_y1 - 75:
                        continue
                else:
                    if cy > line_y2 + 75:
                        continue
                
                if track_id not in tracked:
                    tracked[track_id] = frame_number
                
                # Track entry and exit frames
                if direction == "bottom_to_top":
                    if cy == line_y2:
                        firstLineFrame[track_id] = frame_number
                    elif track_id in firstLineFrame and cy == line_y1:
                        temp = firstLineFrame[track_id]
                        print(f"tracked: {tracked.pop(track_id)}", flush=True)
                        print(f"First: {temp}", flush=True)
                        print(f"Last:  {frame_number}", flush=True)
                        delta_frames = abs(frame_number - temp)
                        delta_time = delta_frames / fps
                        speed = (distance_m / delta_time) * 3.6
                        log = f"Vehicle {track_id} - Speed: {speed:.2f} km/h (bottom_to_top) - Time: {frame_number / 25:.2f}s"
                        speed_log.append(log)
                        print(f"🚗 {log}", flush=True)
                else:
                    if cy == line_y1:
                        firstLineFrame[track_id] = frame_number
                    elif track_id in firstLineFrame and cy == line_y2:
                        temp = firstLineFrame[track_id]
                        print(f"tracked: {tracked.pop(track_id)}", flush=True)
                        print(f"First: {temp}", flush=True)
                        print(f"Last:  {frame_number}", flush=True)
                        delta_frames = abs(frame_number - temp)
                        delta_time = delta_frames / fps
                        speed = (distance_m / delta_time) * 3.6
                        log = f"Vehicle {track_id} - Speed: {speed:.2f} km/h (top_to_bottom) - Time: {frame_number / 25:.2f}s"
                        speed_log.append(log)
                        print(f"🚗 {log}", flush=True)

        # --- LOGGING ---
        log_header = f"[{datetime.datetime.utcnow().isoformat()}] Processed: {video_name}\n"
        log_line = log_header + "\n".join(speed_log) + "\n"
        log_filename = f"{video_name}.log"
        local_log = f"/tmp/{log_filename}"

        with open(local_log, "w") as f:
            f.write(log_line)

        blob_service = BlobServiceClient.from_connection_string(storage_conn)
        blob_client = blob_service.get_blob_client(container="test-logs", blob=log_filename)

        with open(local_log, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        print(f"📝 Uploaded log to 'test-logs/{log_filename}'", flush=True)

        await partition_context.update_checkpoint(event)

    except Exception as e:
        print(f"❌ Error in on_event: {e}", flush=True)

async def main():
    print("🟢 Connecting to Event Hub...", flush=True)

    client = EventHubConsumerClient.from_connection_string(
        conn_str=conn_str,
        consumer_group="$Default",
        eventhub_name="video-segments",
        checkpoint_store=checkpoint_store
    )

    print("🟢 Connected! Listening for events...", flush=True)

    async with client:
       await client.receive(
        on_event=on_event
    )

if __name__ == "__main__":
    asyncio.run(main())
