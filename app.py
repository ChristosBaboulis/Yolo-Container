print("🚀 script started", flush=True)
import asyncio
from azure.eventhub.aio import EventHubConsumerClient
import os
from azure.storage.blob import BlobServiceClient
import datetime
from azure.eventhub.extensions.checkpointstoreblobaio import BlobCheckpointStore
import cv2
from ultralytics import YOLO
import json
print("🔧 imports done", flush=True)

conn_str = os.getenv("EVENT_HUB_CONNECTION", "")
storage_conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
checkpoint_store = BlobCheckpointStore.from_connection_string(
    storage_conn, "eventhub-checkpoints"
)

print("🔧 conn_str done", flush=True)

async def on_event(partition_context, event):
    try:
        print(f"📩 Received on partition {partition_context.partition_id}", flush=True)

        #------------------------------------------------------------------- EXTRACT VIDEO NAME FROM TOPIC -------------------------------------------------------------------
        video_name_raw = event.body_as_str()
        #print(f"📦 Message Raw: {video_name_raw}", flush=True)
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
        #------------------------------------------------------------------- EXTRACT VIDEO NAME FROM TOPIC -------------------------------------------------------------------

        # -------------------------------------------------------------------------- DOWNLOAD VIDEO ---------------------------------------------------------------------------
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
        # -------------------------------------------------------------------------- DOWNLOAD VIDEO ---------------------------------------------------------------------------

        # ------------------------------------------------------------------- OPEN VIDEO - APPLY ANALYSIS --------------------------------------------------------------------
        cap = cv2.VideoCapture(local_video)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_name}", flush=True)
            return

        model = YOLO("yolov8n.pt")
        trackers = {}
        next_id = 0
        speed_log = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        vehicle_classes = ['car', 'truck', 'motorbike']
        distance_m = 20
        line_y1 = 415
        line_y2 = 555

        left_lane_count = 0
        right_lane_count = 0
        speeding_violations = 0
        total_speed_left = 0.0
        total_speed_right = 0.0
        vehicle_logs = []

        # ----------------------- WHILE PER FRAME OF VIDEO -----------------------
        while True:
            # READ FRAME
            ret, frame = cap.read()
            if not ret:
                break
            
            # GET CURRENT FRAME
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # SKIP FIRST SECOND OF THE VIDEO
            if frame_number < 25:
                continue

            # UPDATE BOX TRACKERS IF THEY HAVE BEEN CREATED
            for obj_id in list(trackers.keys()):
                data = trackers[obj_id]
                success, bbox = data["tracker"].update(frame)
                if not success:
                    del trackers[obj_id]
                    continue

                # BOX COORDINATES
                x, y, w, h = map(int, bbox)
                cx = x + w // 2
                cy = y + h // 2
                direction = data["direction"]

                # DELETE BOX IF PASSED THE LINES OF INTEREST
                if direction == "top_to_bottom" and ((y + h > line_y2 + 75) or x >= frame_width - 50 ):
                    del trackers[obj_id]
                    continue
                elif direction == "bottom_to_top" and y < line_y1 - 75:
                    del trackers[obj_id]
                    continue

                # DELETE BOX IF LEFTOVER
                if "speed" not in data:
                    if direction == "top_to_bottom" and (y > line_y2):
                        del trackers[obj_id]
                        continue
                    elif direction == "bottom_to_top" and y + h < line_y1:
                        del trackers[obj_id]
                        continue

                # UPDATE POSITIONS AND FRAMES
                data["positions"].append((cx, cy))
                data["frames"].append(frame_number)

                # DETECT CROSSING LINES
                if len(data["positions"]) >= 2:
                    prev_y = data["positions"][-2][1]
                    curr_y = data["positions"][-1][1]
                    if direction == "top_to_bottom":
                        if data["start"] is None and prev_y < line_y1 <= curr_y:
                            data["start"] = frame_number
                        elif data["end"] is None and prev_y < line_y2 <= curr_y:
                            data["end"] = frame_number
                    elif direction == "bottom_to_top":
                        if data["start"] is None and prev_y > line_y2 >= curr_y:
                            data["start"] = frame_number
                        elif data["end"] is None and prev_y > line_y1 >= curr_y:
                            data["end"] = frame_number

                    # CALCULATE SPEED
                    if data["start"] and data["end"] and "speed" not in data:
                        delta_frames = abs(data["end"] - data["start"])
                        delta_time = delta_frames / fps
                        speed = (distance_m / delta_time) * 3.6
                        data["speed"] = round(speed, 2)

                        # --- Timestamp Calculation ---
                        frame_of_speed = data["end"]
                        seconds_local = frame_of_speed / fps

                        segment_number = int(video_name.split("_")[1].split(".")[0])  # eg 1 for segment_001
                        global_seconds = seconds_local + (segment_number * 120)

                        minutes = int(global_seconds // 60)
                        seconds = int(global_seconds % 60)
                        timestamp_str = f"{minutes:02d}:{seconds:02d}"
                        # -----------------------------

                        # Unique global id for each segment, eg '0011, 0012, 0013,..' from 'segment_001.mp4'
                        full_id = f"{segment_number}{obj_id:02d}"

                        vehicle_logs.append(
                            f"ID: {full_id} | Type: {data['class']} | Direction: {'Right' if direction == 'top_to_bottom' else 'Left'} | Speed: {data['speed']} km/h | Time: {timestamp_str}"
                        )

                        if direction == "top_to_bottom":
                            total_speed_right += data["speed"]
                        else:
                            total_speed_left += data["speed"]

                        # CHECK IF SPEED VIOLATION HAPPENS
                        if (data["class"] == "car" and data["speed"] > 90) or \
                        (data["class"] == "truck" and data["speed"] > 80):
                            speeding_violations += 1

                            # LOG PROGRESS EVERY 10 VIOLATIONS (TO BE DELETED)
                            #if speeding_violations % 10 == 0:
                                #print(f"🚨 Progress: {speeding_violations} vehicles over speed limit", flush=True)

                        #REAL TIME ALERT
                        if data["speed"] > 130:
                            print(f"⚠️ REAL-TIME ALERT: ID:{obj_id} | {data['class']} | Dir: {direction} | {data['speed']} km/h", flush=True)

                        #LOG EVERY CARS SPEED FOR DEBUGGINTG (TO BE DELETED)
                        #print(f"🚗 ID:{obj_id} | Class: {data['class']} | Dir: {direction} | Speed: {data['speed']} km/h", flush=True)

                        #LOG PROGRESS ON COUNTING CARS ON BOTH SIDES (TO BE DELETED)
                        if direction == "top_to_bottom":
                            right_lane_count += 1
                        else:
                            left_lane_count += 1

            #DETECT NEW VEHICLES EVERY 10 FRAMES (~0.4 SECONDS)
            if frame_number % 10 == 0:
                frame_time_sec = frame_number / fps
                minutes = int(frame_time_sec // 60)
                seconds = int(frame_time_sec % 60)

                #DETECT CARS
                detections = model(frame, verbose=False)[0]
                for r in detections.boxes:
                    #DETECT AND SKIP IF NOT IN vehicle_classes
                    cls_id = int(r.cls.item())
                    class_name = model.names[cls_id]
                    if class_name not in vehicle_classes:
                        continue

                    x1, y1, x2, y2 = map(int, r.xyxy[0])
                    w = x2 - x1
                    h = y2 - y1
                    cx = x1 + w // 2

                    #----------------------------------------------------
                    # EXTRA CHECK BECAUSE SOMETIMES DETECTS SAME VEHICLE TWICE,
                    # SO IF THERE IS ALREADY A BOX NEARBY, SKLIP DETECTION
                    #  
                    # DISTINGUISH THRESHOLD BETWEEN CAR, TRUCK
                    threshold = 50 if class_name != "truck" else 100

                    # CHECK IF THERE IS A BOX NEARBY (IN THRESHOLD)
                    already_tracked = False
                    for t in trackers.values():
                        px, py = t["positions"][-1]
                        if abs(cx - px) < threshold:
                            already_tracked = True
                            break
                    #----------------------------------------------------

                    # CREATE NEW TRACKER BOX
                    if not already_tracked:
                        # DISMISS IF IT IS OUTSIDE OF INTEREST LINE
                        if cx < frame_width // 2 and y1 < line_y1:
                            continue

                        # DISMISS IF PASSED THE LINES
                        if cx >= frame_width // 2 and (
                            y2 > line_y2 + 100 or
                            cx > frame_width - 100
                        ):
                            continue

                        # DISMISS IF BOX IS TOO BIG (SOMETIMES DETECTS HUGE BOX WHICH INTERFEERS WITH THE REST DETECTIONS)
                        if w > 200 or h > 200:
                            continue

                        direction = "bottom_to_top" if cx < frame_width // 2 else "top_to_bottom"
                        if direction == "bottom_to_top" and y2 < line_y2:
                            continue

                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(frame, (x1, y1, w, h))
                        trackers[next_id] = {
                            "tracker": tracker,
                            "positions": [(cx, y1 + h // 2)],
                            "frames": [frame_number],
                            "birth": frame_number,
                            "class": class_name,
                            "direction": direction,
                            "start": None,
                            "end": None
                        }
                        next_id += 1
        # ------------------------------------------------------------------- OPEN VIDEO - APPLY ANALYSIS --------------------------------------------------------------------

        # ------------------------------------------------------------------------------- LOGGING ----------------------------------------------------------------------------
        log_header = f"[{datetime.datetime.utcnow().isoformat()}] Processed: {video_name}\n"
        log_body = "\n".join(speed_log)
        avg_speed_left = total_speed_left / left_lane_count if left_lane_count > 0 else 0
        avg_speed_right = total_speed_right / right_lane_count if right_lane_count > 0 else 0

        summary = f"\nTotal vehicles: Left = {left_lane_count} | Right = {right_lane_count}"
        summary += f"\nTotal speed violations: {speeding_violations}"
        summary += f"\nAverage speed: Left = {avg_speed_left:.2f} km/h | Right = {avg_speed_right:.2f} km/h"

        details_section = "\n\n=== Vehicle Details ===\n" + "\n".join(vehicle_logs)

        log_line = log_header + log_body + summary + details_section + "\n"
        
        log_filename = f"{video_name}.log"
        local_log = f"/tmp/{log_filename}"

        with open(local_log, "w") as f:
            f.write(log_line)

        blob_service = BlobServiceClient.from_connection_string(storage_conn)
        blob_client = blob_service.get_blob_client(container="test-logs", blob=log_filename)

        with open(local_log, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        print(f"📝 Uploaded log to 'test-logs/{log_filename}'", flush=True)
        # ------------------------------------------------------------------------------- LOGGING ----------------------------------------------------------------------------

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
