import cv2
import mediapipe as mp
from scipy.spatial import distance
from mediapipe.tasks.python import vision

def eye_aspect_ratio(eye_points):
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    if C == 0:
        return 0
    return (A + B) / (2.0 * C)

def calculate_blink_rate(video_path):
        # Create a fresh detector for each video
    base_options = mp.tasks.BaseOptions(
        model_asset_path="face_landmarker_v2_with_blendshapes.task"
    )
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # fallback if FPS not read properly
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps

    eye_ar_thresh = 0.21  # slightly higher for better detection
    eye_ar_consec_frames = 3

    counter = 0
    total_blinks = 0

    left_eye_idx = [33, 160, 158, 133, 153, 144]
    right_eye_idx = [362, 385, 387, 263, 373, 380]

    frame_number = 0  # used to calculate timestamp

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # convert frame to RGB for mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # calculate timestamp for this frame
        frame_timestamp_ms = int(frame_number * 1000 / fps)
        result = detector.detect_for_video(mp_image, frame_timestamp_ms)
        frame_number += 1

        if not result.face_landmarks:
            continue

        face_landmarks = result.face_landmarks[0]
        h, w, _ = frame.shape

        # get the eye landmarks
        left_eye = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in left_eye_idx]
        right_eye = [(int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)) for i in right_eye_idx]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # blink detection
        if ear < eye_ar_thresh:
            counter += 1
        else:
            if counter >= eye_ar_consec_frames:
                total_blinks += 1
            counter = 0

    cap.release()

    blink_rate = total_blinks / (duration / 60)  # blinks per minute
    return blink_rate, total_blinks

# calculate blink rate for both videos
rate_movie, total_movie = calculate_blink_rate("movie.mp4")
rate_reading, total_reading = calculate_blink_rate("document_video.mp4")

difference = abs(rate_movie - rate_reading)

# print results
print("Video 1 (Movie) Blinks:", total_movie)
print("Video 1 Blink Rate:", rate_movie)

print("Video 2 (Reading) Blinks:", total_reading)
print("Video 2 Blink Rate:", rate_reading)

print("Difference in Blink Rate:", difference)