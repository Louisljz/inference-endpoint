import mediapipe as mp
import numpy as np
import cv2


class VideoProcessor:
    N_FRAMES = 113  # number of frames to sample from the video
    N_LANDMARKS = 33 + 21 + 21  # number of landmarks for pose, left hand, and right hand

    # Define keypoints for angle calculation (indices start from 0)
    POSE_ANGLE_INDICES = [
        (12, 14, 16),
        (14, 16, 18),
        (18, 16, 22),
        (14, 12, 24),
        (11, 13, 15),
        (13, 15, 17),
        (17, 15, 21),
        (13, 11, 23),
    ]

    # For both left and right hands
    HAND_ANGLE_INDICES = [
        (4, 0, 8),
        (8, 0, 16),
        (0, 9, 12),
        (0, 17, 20),
    ]

    def __init__(self):
        self.model = mp.solutions.holistic.Holistic(static_image_mode=False,
                                                  min_detection_confidence=0.3,
                                                  min_tracking_confidence=0.3)

    def motion_trim(self, video_path: str) -> np.ndarray:
        try:
            cap = cv2.VideoCapture(video_path)

            # Pre-allocate memory for frames
            frames = []
            motion_scores = []

            # Read frames in batches
            prev_frame = None
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (1280, 720))
                frames.append(frame)

                # Calculate motion score on the fly
                if prev_frame is not None:
                    diff = cv2.absdiff(prev_frame, frame)
                    motion_scores.append(np.mean(diff))

                prev_frame = frame.copy()

            cap.release()

            if not frames:
                return None

            # Trim based on motion
            threshold = np.mean(motion_scores) * 0.3
            start_idx = next(
                (i for i, score in enumerate(motion_scores) if score > threshold), 0
            )
            end_idx = len(frames) - next(
                (i for i, score in enumerate(reversed(motion_scores)) if score > threshold),
                0,
            )

            # Apply trimming
            frames = frames[max(0, start_idx - 5) : min(len(frames), end_idx + 5)]

            # Standardize length
            if frames:
                indices = np.linspace(0, len(frames) - 1, self.N_FRAMES, dtype=int)
                frames = [frames[i] for i in indices]
                return frames

            return None

        except Exception as e:
            print(f"Error processing video {video_path}: {str(e)}")
            return None

    def extract_landmarks(self, frames: list[np.ndarray]) -> np.ndarray:
        keypoints_sequence = np.zeros((self.N_FRAMES, self.N_LANDMARKS, 3))

        for i, frame in enumerate(frames):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model.process(frame_rgb)

            # Initialize keypoints arrays
            pose_keypoints = np.zeros((33, 3))
            left_hand_keypoints = np.zeros((21, 3))
            right_hand_keypoints = np.zeros((21, 3))

            # Extract pose landmarks if detected
            if results.pose_landmarks:
                pose_keypoints = np.array([[lm.x, lm.y, lm.z]
                                         for lm in results.pose_landmarks.landmark])

            # Extract hand landmarks if detected
            if results.left_hand_landmarks:
                left_hand_keypoints = np.array([[lm.x, lm.y, lm.z]
                                              for lm in results.left_hand_landmarks.landmark])
            if results.right_hand_landmarks:
                right_hand_keypoints = np.array([[lm.x, lm.y, lm.z]
                                               for lm in results.right_hand_landmarks.landmark])

            # Combine all keypoints
            frame_keypoints = np.concatenate([
                pose_keypoints,
                left_hand_keypoints,
                right_hand_keypoints
            ])
            keypoints_sequence[i] = frame_keypoints

        return keypoints_sequence

    @staticmethod
    def calculate_angle(A, B, C):
        BA = A - B
        BC = C - B
        # Compute dot product and magnitudes
        dot_product = np.dot(BA, BC)
        magnitude_BA = np.linalg.norm(BA)
        magnitude_BC = np.linalg.norm(BC)
        # Prevent division by zero
        if magnitude_BA == 0 or magnitude_BC == 0:
            return 0.0
        # Calculate the cosine of the angle
        cos_angle = dot_product / (magnitude_BA * magnitude_BC)
        # Clip values to handle numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        # Return the angle in radians
        return np.arccos(cos_angle)

    def get_angles_from_video(self, keypoints_sequence: np.ndarray) -> np.ndarray:
        video_angles = []
        for frame in keypoints_sequence:
            frame_angles = []

            # Pose angles
            for i, j, k in self.POSE_ANGLE_INDICES:
                frame_angles.append(self.calculate_angle(frame[i], frame[j], frame[k]))

            # Left hand angles
            for i, j, k in self.HAND_ANGLE_INDICES:
                frame_angles.append(
                    self.calculate_angle(frame[33 + i], frame[33 + j], frame[33 + k])
                )

            # Right hand angles
            for i, j, k in self.HAND_ANGLE_INDICES:
                frame_angles.append(
                    self.calculate_angle(frame[54 + i], frame[54 + j], frame[54 + k])
                )

            video_angles.append(frame_angles)

        return np.array(video_angles)

    def process_video(self, video_path: str) -> np.ndarray:
        frames = self.motion_trim(video_path)
        landmarks = self.extract_landmarks(frames)        
        angles = self.get_angles_from_video(landmarks) # (113, 16)

        # reshape landmarks from (113, 75, 3) to (113, 225)
        landmarks = landmarks.reshape(self.N_FRAMES, self.N_LANDMARKS * 3)
        model_input = np.concatenate([landmarks, angles], axis=1)
        return model_input


if __name__ == "__main__":
    import time

    processor = VideoProcessor()

    start_time = time.time()
    frames = processor.motion_trim("videos/test_buka.mp4")
    motion_trim_time = time.time() - start_time
    print(f"Motion trim took {motion_trim_time:.2f} seconds")

    start_time = time.time()
    landmarks = processor.extract_landmarks(frames)
    landmark_time = time.time() - start_time
    print(f"Landmark extraction took {landmark_time:.2f} seconds")

    start_time = time.time()
    angles = processor.get_angles_from_video(landmarks)
    angles_time = time.time() - start_time
    print(f"Angle calculation took {angles_time:.2f} seconds")

    print(f"Total processing time: {motion_trim_time + landmark_time + angles_time:.2f} seconds")
    print('Landmarks shape:', landmarks.shape)
    print('Angles shape:', angles.shape)
