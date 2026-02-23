import cv2, mediapipe as mp, numpy as np
from scipy.spatial import distance
from mediapipe.tasks.python import vision

def eye_aspect_ratio(eye_points):
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    if C == 0:
        return 0
    return (A + B) / (2.0 * C)


def calculate_blink_rate(video_path, detector):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps
    frame_time_increment = int(1000 / fps)
    frame_timestamp_ms = 0

    eye_ar_thresh = 0.21
    eye_ar_consec_frames = 3
    counter = 0
    total = 0

    left_eye_idx = [33, 160, 158, 133, 153, 144]
    right_eye_idx = [362, 385, 387, 263, 373, 380]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = detector.detect_for_video(mp_image, frame_timestamp_ms)
        frame_timestamp_ms += frame_time_increment

        if not result.face_landmarks:
            continue

        face_landmarks = result.face_landmarks[0]
        h, w, _ = frame.shape

        left_eye = [(int(face_landmarks[i].x * w),
                     int(face_landmarks[i].y * h)) for i in left_eye_idx]

        right_eye = [(int(face_landmarks[i].x * w),
                      int(face_landmarks[i].y * h)) for i in right_eye_idx]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        if ear < eye_ar_thresh:
            counter += 1
        else:
            if counter >= eye_ar_consec_frames:
                total += 1
            counter = 0

    cap.release()

    blink_rate = total / (duration / 60)
    return blink_rate, total


# Now run for 2 videos

base_options = mp.tasks.BaseOptions(
    model_asset_path="face_landmarker_v2_with_blendshapes.task"
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)

rate1, total1 = calculate_blink_rate("movie1.mp4", detector)
rate2, total2 = calculate_blink_rate("movie2.mp4", detector)

difference = rate1 - rate2

print("Video 1 Blinks:", total1)
print("Video 1 Blink Rate:", rate1)

print("Video 2 Blinks:", total2)
print("Video 2 Blink Rate:", rate2)

print("Difference in Blink Rate:", difference)