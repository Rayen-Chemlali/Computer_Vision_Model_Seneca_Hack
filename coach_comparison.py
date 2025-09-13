import cv2
import numpy as np
import json
import os
from typing import List, Dict, Optional
from utils import PoseUtils
from datetime import datetime

class CoachVideoAnalyzer:
    def __init__(self):
        self.pose_utils = PoseUtils()
        self.coach_data_dir = "coach_data"
        os.makedirs(self.coach_data_dir, exist_ok=True)
    
    def process_local_coach_video(self, video_path: str, exercise_type: str) -> bool:
        """Process a local coach video for the specified exercise"""
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return False
        
        print(f"Processing coach video for {exercise_type} from: {video_path}")
        
        # Process video and extract landmarks
        print(f"Extracting pose landmarks...")
        landmarks_sequence = self.pose_utils.extract_landmarks_from_video(video_path, max_frames=200)
        
        if not landmarks_sequence:
            print("No pose landmarks detected in coach video")
            return False
        
        # Save coach data
        coach_data = {
            'exercise_type': exercise_type,
            'video_path': video_path,
            'total_frames': len(landmarks_sequence),
            'landmarks_sequence': [landmarks.tolist() for landmarks in landmarks_sequence],
            'processed_date': datetime.now().isoformat()
        }
        
        coach_file = os.path.join(self.coach_data_dir, f"{exercise_type}_coach.json")
        with open(coach_file, 'w') as f:
            json.dump(coach_data, f)
        
        print(f"Coach data saved: {len(landmarks_sequence)} frames processed")
        return True
    
    def real_time_coach_comparison(self, exercise_type: str, camera_index: int = 0) -> None:
        """Real-time comparison with coach video"""
        # Load coach data
        coach_file = os.path.join(self.coach_data_dir, f"{exercise_type}_coach.json")
        if not os.path.exists(coach_file):
            print(f"Coach data not found for {exercise_type}. Please process a coach video first.")
            return
        
        with open(coach_file, 'r') as f:
            coach_data = json.load(f)
        
        coach_landmarks = [np.array(landmarks) for landmarks in coach_data['landmarks_sequence']]
        print(f"Loaded coach data: {len(coach_landmarks)} frames")
        
        # Start webcam
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Cannot open camera {camera_index}")
            return
        
        print("Starting real-time comparison...")
        print("Instructions:")
        print("- Position yourself in frame")  
        print("- Press 'g' when ready to start exercise")
        print("- Press 's' to stop comparison and save session")
        print("- Press 'q' to quit")
        
        user_sequence = []
        comparing = False
        comparison_start_frame = 0
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract landmarks
            landmarks = self.pose_utils.extract_landmarks_from_frame(frame)
            annotated_frame = frame.copy()
            
            if landmarks is not None:
                # Draw pose landmarks
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose_utils.pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    self.pose_utils.mp_drawing.draw_landmarks(
                        annotated_frame, results.pose_landmarks,
                        self.pose_utils.mp_pose.POSE_CONNECTIONS
                    )
                
                # If comparing, add to sequence and compare
                if comparing:
                    user_sequence.append(landmarks)
                    current_comparison_frame = len(user_sequence) - 1
                    
                    # Compare with coach frame (with some tolerance)
                    if current_comparison_frame < len(coach_landmarks):
                        feedback = self._compare_poses(landmarks, coach_landmarks[current_comparison_frame])
                        
                        # Display feedback
                        cv2.putText(annotated_frame, feedback, (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Show progress
                        progress = f"Frame: {current_comparison_frame + 1}/{len(coach_landmarks)}"
                        cv2.putText(annotated_frame, progress, (10, 90),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    else:
                        cv2.putText(annotated_frame, "Exercise completed!", (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Status display
            status = "COMPARING" if comparing else "READY - Press 'g' to start"
            cv2.putText(annotated_frame, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow(f"Coach Comparison - {exercise_type}", annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('g') and not comparing:
                print("Starting comparison!")
                comparing = True
                comparison_start_frame = frame_count
                user_sequence = []
            elif key == ord('s') and comparing:
                print("Stopping comparison")
                comparing = False
                if user_sequence:
                    self._save_user_session(user_sequence, exercise_type)
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Final analysis
        if user_sequence and len(user_sequence) > 10:
            print("\nFinal analysis:")
            overall_score = self._analyze_full_sequence(user_sequence, coach_landmarks)
            print(f"Overall performance: {overall_score:.1f}/10")
    
    def _compare_poses(self, user_landmarks: np.ndarray, coach_landmarks: np.ndarray) -> str:
        """Compare single pose frames and return feedback"""
        user_reshaped = user_landmarks.reshape(33, 4)
        coach_reshaped = coach_landmarks.reshape(33, 4)
        
        # Calculate key joint angles
        user_angles = self.pose_utils.calculate_joint_angles_from_landmarks(user_landmarks)
        coach_angles = self.pose_utils.calculate_joint_angles_from_landmarks(coach_landmarks)
        
        feedback_items = []
        
        # Compare key angles
        for joint in ['left_knee', 'right_knee', 'left_elbow', 'right_elbow']:
            if joint in user_angles and joint in coach_angles:
                user_angle = user_angles[joint]
                coach_angle = coach_angles[joint]
                diff = abs(user_angle - coach_angle)
                
                if diff > 20:  # More than 20 degrees difference
                    if 'knee' in joint:
                        if user_angle < coach_angle:
                            feedback_items.append(f"Bend {joint.replace('_', ' ')} more")
                        else:
                            feedback_items.append(f"Straighten {joint.replace('_', ' ')}")
                    elif 'elbow' in joint:
                        if user_angle < coach_angle:
                            feedback_items.append(f"Extend {joint.replace('_', ' ')}")
                        else:
                            feedback_items.append(f"Bend {joint.replace('_', ' ')}")
        
        if not feedback_items:
            return "Good form!"
        
        return feedback_items[0]  # Return first feedback
    
    def _analyze_full_sequence(self, user_sequence: List[np.ndarray], coach_sequence: List[np.ndarray]) -> float:
        """Analyze complete sequence and return score"""
        if not user_sequence or not coach_sequence:
            return 0.0
        
        # Normalize sequences
        user_normalized = self.pose_utils.normalize_pose_sequence(user_sequence)
        coach_normalized = self.pose_utils.normalize_pose_sequence(coach_sequence)
        
        # Calculate similarity
        dtw_distance = self.pose_utils.compare_sequences_dtw(user_normalized, coach_normalized)
        
        # Convert to score (lower distance = higher score)
        score = max(0, 10 - (dtw_distance * 5))
        return min(10, score)
    
    def _save_user_session(self, user_sequence: List[np.ndarray], exercise_type: str) -> None:
        """Save user session data"""
        session_data = {
            'exercise_type': exercise_type,
            'total_frames': len(user_sequence),
            'landmarks_sequence': [landmarks.tolist() for landmarks in user_sequence],
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f"user_session_{exercise_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(session_data, f)
        
        print(f"User session saved: {filename}")

    def list_processed_exercises(self) -> List[str]:
        """List exercises that have been processed and are available for comparison"""
        available = []
        for filename in os.listdir(self.coach_data_dir):
            if filename.endswith('_coach.json'):
                exercise_type = filename.replace('_coach.json', '')
                available.append(exercise_type)
        return available

# Main functions to be called from main.py
def process_coach_video(video_path: str, exercise_type: str) -> bool:
    """Process a local coach video"""
    analyzer = CoachVideoAnalyzer()
    return analyzer.process_local_coach_video(video_path, exercise_type)

def start_coach_comparison(exercise_type: str, camera_index: int = 0) -> None:
    """Start real-time coach comparison"""
    analyzer = CoachVideoAnalyzer()
    analyzer.real_time_coach_comparison(exercise_type, camera_index)

def list_available_exercises() -> List[str]:
    """List processed exercises available for comparison"""
    analyzer = CoachVideoAnalyzer()
    return analyzer.list_processed_exercises()
