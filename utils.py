import cv2
import mediapipe as mp
import numpy as np
import json
import pandas as pd
from scipy.spatial.distance import euclidean, cosine
from scipy.optimize import linear_sum_assignment
from dtaidistance import dtw
import os
from typing import List, Dict, Tuple, Optional, Union
import pickle
from datetime import datetime


class PoseUtils:
    """Shared utilities for pose detection and comparison across all methods"""

    def __init__(self):
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # MediaPipe landmark indices for key body parts
        self.POSE_LANDMARKS = {
            'nose': 0,
            'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
            'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
            'left_ear': 7, 'right_ear': 8,
            'mouth_left': 9, 'mouth_right': 10,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_pinky': 17, 'right_pinky': 18,
            'left_index': 19, 'right_index': 20,
            'left_thumb': 21, 'right_thumb': 22,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
            'left_heel': 29, 'right_heel': 30,
            'left_foot_index': 31, 'right_foot_index': 32
        }

        # Important joint groups for exercise analysis
        self.JOINT_GROUPS = {
            'upper_body': [11, 12, 13, 14, 15, 16],  # shoulders, elbows, wrists
            'lower_body': [23, 24, 25, 26, 27, 28],  # hips, knees, ankles
            'core': [11, 12, 23, 24],  # shoulders and hips for posture
            'arms': [13, 14, 15, 16],  # elbows and wrists
            'legs': [25, 26, 27, 28]  # knees and ankles
        }

    def extract_landmarks_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract 33 pose landmarks from a single frame"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                return np.array(landmarks)  # Shape: (132,) = 33 landmarks * 4 values
            return None
        except Exception as e:
            print(f"Error extracting landmarks: {e}")
            return None

    def extract_landmarks_from_video(self, video_path: str, max_frames: int = None) -> List[np.ndarray]:
        """Extract landmarks from entire video"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        landmarks_sequence = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames and frame_count >= max_frames:
                break

            landmarks = self.extract_landmarks_from_frame(frame)
            if landmarks is not None:
                landmarks_sequence.append(landmarks)

            frame_count += 1

        cap.release()
        print(f"Extracted landmarks from {len(landmarks_sequence)} frames")
        return landmarks_sequence

    def save_landmarks_sequence(self, landmarks_sequence: List[np.ndarray],
                                filename: str, format: str = 'json') -> None:
        """Save landmarks sequence to file"""
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)

        if format == 'json':
            # Convert numpy arrays to lists for JSON serialization
            data = {
                'landmarks_sequence': [landmarks.tolist() for landmarks in landmarks_sequence],
                'num_frames': len(landmarks_sequence),
                'landmark_format': '33 landmarks * 4 values (x, y, z, visibility)',
                'timestamp': datetime.now().isoformat()
            }
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)

        elif format == 'numpy':
            # Stack into 3D array: (frames, landmarks, values)
            stacked = np.stack(landmarks_sequence)
            np.save(filename, stacked)

        elif format == 'pickle':
            with open(filename, 'wb') as f:
                pickle.dump({
                    'landmarks_sequence': landmarks_sequence,
                    'timestamp': datetime.now().isoformat()
                }, f)

        print(f"Saved landmarks sequence to {filename} ({format} format)")

    def load_landmarks_sequence(self, filename: str, format: str = 'auto') -> List[np.ndarray]:
        """Load landmarks sequence from file"""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File not found: {filename}")

        # Auto-detect format from extension
        if format == 'auto':
            ext = os.path.splitext(filename)[1].lower()
            format = {'json': 'json', '.npy': 'numpy', '.pkl': 'pickle', '.pickle': 'pickle'}.get(ext, 'json')

        if format == 'json':
            with open(filename, 'r') as f:
                data = json.load(f)
            return [np.array(landmarks) for landmarks in data['landmarks_sequence']]

        elif format == 'numpy':
            data = np.load(filename)
            return [data[i] for i in range(len(data))]

        elif format == 'pickle':
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            return data['landmarks_sequence']

    def normalize_pose_sequence(self, landmarks_sequence: List[np.ndarray],
                                method: str = 'shoulder_distance') -> List[np.ndarray]:
        """Normalize pose sequences to account for different body sizes and distances from camera"""
        normalized_sequence = []

        for landmarks in landmarks_sequence:
            # Reshape to (33, 4) for easier manipulation
            reshaped = landmarks.reshape(33, 4)

            if method == 'shoulder_distance':
                # Use shoulder distance as normalization factor
                left_shoulder = reshaped[11, :2]  # x, y only
                right_shoulder = reshaped[12, :2]
                shoulder_dist = np.linalg.norm(right_shoulder - left_shoulder)

                if shoulder_dist > 0:
                    # Normalize x, y coordinates by shoulder distance
                    reshaped[:, :2] = reshaped[:, :2] / shoulder_dist

            elif method == 'hip_distance':
                # Use hip distance as normalization factor
                left_hip = reshaped[23, :2]
                right_hip = reshaped[24, :2]
                hip_dist = np.linalg.norm(right_hip - left_hip)

                if hip_dist > 0:
                    reshaped[:, :2] = reshaped[:, :2] / hip_dist

            # Center the pose (subtract mean position)
            mean_pos = np.mean(reshaped[:, :2], axis=0)
            reshaped[:, :2] -= mean_pos

            normalized_sequence.append(reshaped.flatten())

        return normalized_sequence

    def calculate_joint_angles_from_landmarks(self, landmarks: np.ndarray) -> Dict[str, float]:
        """Calculate joint angles from landmarks"""
        reshaped = landmarks.reshape(33, 4)
        angles = {}

        def angle_between_points(p1, p2, p3):
            """Calculate angle at p2 formed by p1-p2-p3"""
            v1 = p1 - p2
            v2 = p3 - p2
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

        # Key joint angles
        try:
            # Left elbow: shoulder - elbow - wrist
            angles['left_elbow'] = angle_between_points(
                reshaped[11, :2], reshaped[13, :2], reshaped[15, :2])

            # Right elbow
            angles['right_elbow'] = angle_between_points(
                reshaped[12, :2], reshaped[14, :2], reshaped[16, :2])

            # Left knee: hip - knee - ankle
            angles['left_knee'] = angle_between_points(
                reshaped[23, :2], reshaped[25, :2], reshaped[27, :2])

            # Right knee
            angles['right_knee'] = angle_between_points(
                reshaped[24, :2], reshaped[26, :2], reshaped[28, :2])

            # Left hip: shoulder - hip - knee
            angles['left_hip'] = angle_between_points(
                reshaped[11, :2], reshaped[23, :2], reshaped[25, :2])

            # Right hip
            angles['right_hip'] = angle_between_points(
                reshaped[12, :2], reshaped[24, :2], reshaped[26, :2])

        except Exception as e:
            print(f"Error calculating angles: {e}")

        return angles

    def compare_sequences_dtw(self, seq1: List[np.ndarray], seq2: List[np.ndarray]) -> float:
        """Compare two pose sequences using Dynamic Time Warping"""
        if not seq1 or not seq2:
            return float('inf')

        # Convert to 2D arrays for DTW
        seq1_array = np.array(seq1)
        seq2_array = np.array(seq2)

        try:
            # Calculate DTW distance
            distance = dtw.distance(seq1_array, seq2_array)
            return distance
        except Exception as e:
            print(f"DTW comparison error: {e}")
            return float('inf')

    def compare_sequences_euclidean(self, seq1: List[np.ndarray], seq2: List[np.ndarray]) -> float:
        """Compare sequences using average Euclidean distance"""
        if not seq1 or not seq2:
            return float('inf')

        # Time-align sequences by interpolation
        min_len = min(len(seq1), len(seq2))
        if min_len == 0:
            return float('inf')

        # Resample to same length
        seq1_resampled = self._resample_sequence(seq1, min_len)
        seq2_resampled = self._resample_sequence(seq2, min_len)

        # Calculate frame-by-frame Euclidean distance
        distances = []
        for i in range(min_len):
            dist = euclidean(seq1_resampled[i], seq2_resampled[i])
            distances.append(dist)

        return np.mean(distances)

    def compare_sequences_cosine(self, seq1: List[np.ndarray], seq2: List[np.ndarray]) -> float:
        """Compare sequences using cosine similarity"""
        if not seq1 or not seq2:
            return float('inf')

        # Flatten sequences for global comparison
        seq1_flat = np.concatenate(seq1)
        seq2_flat = np.concatenate(seq2)

        # Pad shorter sequence
        if len(seq1_flat) != len(seq2_flat):
            min_len = min(len(seq1_flat), len(seq2_flat))
            seq1_flat = seq1_flat[:min_len]
            seq2_flat = seq2_flat[:min_len]

        try:
            # Calculate cosine distance (1 - cosine similarity)
            return cosine(seq1_flat, seq2_flat)
        except Exception as e:
            print(f"Cosine comparison error: {e}")
            return float('inf')

    def _resample_sequence(self, sequence: List[np.ndarray], target_length: int) -> List[np.ndarray]:
        """Resample sequence to target length using linear interpolation"""
        if len(sequence) == target_length:
            return sequence

        # Create indices for interpolation
        old_indices = np.linspace(0, len(sequence) - 1, len(sequence))
        new_indices = np.linspace(0, len(sequence) - 1, target_length)

        # Interpolate each dimension
        resampled = []
        for i in range(target_length):
            # Find closest frames for interpolation
            idx = new_indices[i]
            if idx == int(idx):  # Exact match
                resampled.append(sequence[int(idx)])
            else:
                # Linear interpolation between two frames
                idx_low = int(np.floor(idx))
                idx_high = int(np.ceil(idx))
                weight = idx - idx_low

                interpolated = (1 - weight) * sequence[idx_low] + weight * sequence[idx_high]
                resampled.append(interpolated)

        return resampled

    def analyze_movement_quality(self, landmarks_sequence: List[np.ndarray]) -> Dict[str, float]:
        """Analyze movement quality metrics"""
        if not landmarks_sequence:
            return {}

        metrics = {}

        # Calculate smoothness (jerk analysis)
        smoothness_scores = []
        for i in range(1, len(landmarks_sequence) - 1):
            prev_landmarks = landmarks_sequence[i - 1].reshape(33, 4)
            curr_landmarks = landmarks_sequence[i].reshape(33, 4)
            next_landmarks = landmarks_sequence[i + 1].reshape(33, 4)

            # Calculate jerk (third derivative of position)
            accel = next_landmarks[:, :2] - 2 * curr_landmarks[:, :2] + prev_landmarks[:, :2]
            jerk = np.mean(np.linalg.norm(accel, axis=1))
            smoothness_scores.append(1 / (1 + jerk))  # Invert so higher is better

        metrics['smoothness'] = np.mean(smoothness_scores) if smoothness_scores else 0

        # Calculate consistency (standard deviation of joint angles)
        all_angles = []
        for landmarks in landmarks_sequence:
            angles = self.calculate_joint_angles_from_landmarks(landmarks)
            all_angles.append(list(angles.values()))

        if all_angles:
            angle_array = np.array(all_angles)
            consistency_scores = []
            for joint_idx in range(angle_array.shape[1]):
                joint_angles = angle_array[:, joint_idx]
                consistency = 1 / (1 + np.std(joint_angles))  # Lower std = higher consistency
                consistency_scores.append(consistency)
            metrics['consistency'] = np.mean(consistency_scores)
        else:
            metrics['consistency'] = 0

        # Calculate movement amplitude
        if len(landmarks_sequence) > 1:
            first_frame = landmarks_sequence[0].reshape(33, 4)
            last_frame = landmarks_sequence[-1].reshape(33, 4)
            displacement = np.mean(np.linalg.norm(last_frame[:, :2] - first_frame[:, :2], axis=1))
            metrics['amplitude'] = displacement
        else:
            metrics['amplitude'] = 0

        return metrics

    def generate_feedback_from_comparison(self, comparison_results: Dict[str, float],
                                          exercise_type: str = 'general') -> List[str]:
        """Generate textual feedback from comparison results"""
        feedback = []

        # Define thresholds for different metrics
        thresholds = {
            'dtw_distance': {'excellent': 0.1, 'good': 0.3, 'needs_improvement': 0.6},
            'euclidean_distance': {'excellent': 0.2, 'good': 0.5, 'needs_improvement': 1.0},
            'cosine_distance': {'excellent': 0.1, 'good': 0.3, 'needs_improvement': 0.6},
            'smoothness': {'excellent': 0.8, 'good': 0.6, 'needs_improvement': 0.4},
            'consistency': {'excellent': 0.8, 'good': 0.6, 'needs_improvement': 0.4}
        }

        for metric, value in comparison_results.items():
            if metric in thresholds:
                thresh = thresholds[metric]

                if metric in ['dtw_distance', 'euclidean_distance', 'cosine_distance']:
                    # Lower is better for distance metrics
                    if value <= thresh['excellent']:
                        feedback.append(f"Excellent form! Your {metric.replace('_', ' ')} is very close to reference.")
                    elif value <= thresh['good']:
                        feedback.append(f"Good form! Your {metric.replace('_', ' ')} shows minor deviations.")
                    elif value <= thresh['needs_improvement']:
                        feedback.append(
                            f"Form needs improvement. Your {metric.replace('_', ' ')} shows significant differences.")
                    else:
                        feedback.append(f"Poor form. Your {metric.replace('_', ' ')} differs greatly from reference.")
                else:
                    # Higher is better for quality metrics
                    if value >= thresh['excellent']:
                        feedback.append(f"Excellent {metric}! Your movement is very smooth and consistent.")
                    elif value >= thresh['good']:
                        feedback.append(f"Good {metric}. Your movement shows good control.")
                    elif value >= thresh['needs_improvement']:
                        feedback.append(f"{metric.capitalize()} needs improvement. Try to move more smoothly.")
                    else:
                        feedback.append(f"Poor {metric}. Focus on controlled, consistent movements.")

        return feedback if feedback else ["Analysis complete. Keep practicing!"]