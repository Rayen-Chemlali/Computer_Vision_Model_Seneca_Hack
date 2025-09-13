import numpy as np
import json
import os
import cv2
from typing import List, Dict
from utils import PoseUtils
from datetime import datetime

class DatasetExerciseAnalyzer:
    def __init__(self):
        self.pose_utils = PoseUtils()
        self.dataset_dir = "datasets"
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        # Create synthetic perfect form datasets
        self._create_synthetic_datasets()
    
    def _create_synthetic_datasets(self) -> None:
        """Create synthetic perfect form exercise datasets"""
        print("Creating synthetic perfect form datasets...")
        
        synthetic_data = {
            'squat': self._generate_perfect_squat(),
            'pushup': self._generate_perfect_pushup(),
            'plank': self._generate_perfect_plank()
        }
        
        for exercise, data in synthetic_data.items():
            filename = os.path.join(self.dataset_dir, f"{exercise}_perfect.json")
            with open(filename, 'w') as f:
                json.dump(data, f)
        
        print("Synthetic datasets created successfully")
    
    def _generate_perfect_squat(self) -> Dict:
        """Generate perfect squat movement data"""
        frames = 60  # 2 seconds at 30fps
        landmarks_sequence = []
        
        for i in range(frames):
            progress = i / frames
            squat_depth = np.sin(progress * np.pi) * 0.3  # 30cm down movement
            
            landmarks = np.zeros(132)  # 33 points * 4 values
            
            # Hip movement (main squat motion)
            landmarks[23*4:23*4+2] = [0.4, 0.6 + squat_depth]  # Left hip
            landmarks[24*4:24*4+2] = [0.6, 0.6 + squat_depth]  # Right hip
            
            # Knee movement
            knee_bend = squat_depth * 1.2
            landmarks[25*4:25*4+2] = [0.4, 0.8 + knee_bend]  # Left knee
            landmarks[26*4:26*4+2] = [0.6, 0.8 + knee_bend]  # Right knee
            
            # Stable ankles
            landmarks[27*4:27*4+2] = [0.4, 0.95]  # Left ankle
            landmarks[28*4:28*4+2] = [0.6, 0.95]  # Right ankle
            
            # Stable shoulders
            landmarks[11*4:11*4+2] = [0.4, 0.3]  # Left shoulder
            landmarks[12*4:12*4+2] = [0.6, 0.3]  # Right shoulder
            
            # Set visibility
            landmarks[3::4] = 1.0
            
            landmarks_sequence.append(landmarks.tolist())
        
        return {
            'exercise_type': 'squat',
            'landmarks_sequence': landmarks_sequence,
            'total_frames': frames,
            'description': 'Perfect squat form reference'
        }
    
    def _generate_perfect_pushup(self) -> Dict:
        """Generate perfect pushup movement data"""
        frames = 40
        landmarks_sequence = []
        
        for i in range(frames):
            progress = i / frames
            pushup_depth = np.sin(progress * np.pi) * 0.15  # 15cm range
            
            landmarks = np.zeros(132)
            
            # Shoulder movement
            landmarks[11*4:11*4+2] = [0.4, 0.4 + pushup_depth]  # Left shoulder
            landmarks[12*4:12*4+2] = [0.6, 0.4 + pushup_depth]  # Right shoulder
            
            # Elbow movement
            landmarks[13*4:13*4+2] = [0.35, 0.5 + pushup_depth]  # Left elbow
            landmarks[14*4:14*4+2] = [0.65, 0.5 + pushup_depth]  # Right elbow
            
            # Wrist movement
            landmarks[15*4:15*4+2] = [0.3, 0.6 + pushup_depth]  # Left wrist
            landmarks[16*4:16*4+2] = [0.7, 0.6 + pushup_depth]  # Right wrist
            
            # Hip follows slightly
            landmarks[23*4:23*4+2] = [0.4, 0.4 + pushup_depth * 0.3]  # Left hip
            landmarks[24*4:24*4+2] = [0.6, 0.4 + pushup_depth * 0.3]  # Right hip
            
            landmarks[3::4] = 1.0
            landmarks_sequence.append(landmarks.tolist())
        
        return {
            'exercise_type': 'pushup',
            'landmarks_sequence': landmarks_sequence,
            'total_frames': frames,
            'description': 'Perfect pushup form reference'
        }
    
    def _generate_perfect_plank(self) -> Dict:
        """Generate perfect plank hold data"""
        frames = 90  # 3 seconds hold
        landmarks_sequence = []
        
        for i in range(frames):
            landmarks = np.zeros(132)
            
            # Static plank position
            # Shoulders
            landmarks[11*4:11*4+2] = [0.4, 0.4]  # Left shoulder
            landmarks[12*4:12*4+2] = [0.6, 0.4]  # Right shoulder
            
            # Elbows (for forearm plank)
            landmarks[13*4:13*4+2] = [0.4, 0.5]  # Left elbow
            landmarks[14*4:14*4+2] = [0.6, 0.5]  # Right elbow
            
            # Hips (straight line)
            landmarks[23*4:23*4+2] = [0.4, 0.4]  # Left hip
            landmarks[24*4:24*4+2] = [0.6, 0.4]  # Right hip
            
            # Knees
            landmarks[25*4:25*4+2] = [0.4, 0.6]  # Left knee
            landmarks[26*4:26*4+2] = [0.6, 0.6]  # Right knee
            
            # Ankles
            landmarks[27*4:27*4+2] = [0.4, 0.8]  # Left ankle
            landmarks[28*4:28*4+2] = [0.6, 0.8]  # Right ankle
            
            landmarks[3::4] = 1.0
            landmarks_sequence.append(landmarks.tolist())
        
        return {
            'exercise_type': 'plank',
            'landmarks_sequence': landmarks_sequence,
            'total_frames': frames,
            'description': 'Perfect plank form reference'
        }
    
    def real_time_dataset_comparison(self, exercise_type: str, camera_index: int = 0) -> None:
        """Real-time comparison with dataset reference"""
        # Load dataset reference
        dataset_file = os.path.join(self.dataset_dir, f"{exercise_type}_perfect.json")
        if not os.path.exists(dataset_file):
            print(f"Dataset not found for {exercise_type}")
            return
        
        with open(dataset_file, 'r') as f:
            dataset_data = json.load(f)
        
        reference_landmarks = [np.array(landmarks) for landmarks in dataset_data['landmarks_sequence']]
        print(f"Loaded dataset reference: {len(reference_landmarks)} frames")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Cannot open camera {camera_index}")
            return
        
        print(f"Starting dataset comparison for {exercise_type}")
        print("Instructions:")
        print("- Press 'g' to start comparison")
        print("- Press 's' to stop and analyze")
        print("- Press 'q' to quit")
        
        user_sequence = []
        comparing = False
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
                
                if comparing:
                    user_sequence.append(landmarks)
                    
                    # Real-time scoring
                    if len(user_sequence) >= 10:
                        recent_score = self._calculate_real_time_score(
                            user_sequence[-10:], reference_landmarks[:10]
                        )
                        cv2.putText(annotated_frame, f"Score: {recent_score:.1f}/10",
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        # Basic feedback
                        feedback = self._get_basic_feedback(recent_score)
                        cv2.putText(annotated_frame, feedback, (10, 90),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Status
            status = "COMPARING" if comparing else "READY - Press 'g' to start"
            cv2.putText(annotated_frame, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow(f"Dataset Comparison - {exercise_type}", annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('g'):
                print("Starting comparison...")
                comparing = True
                user_sequence = []
            elif key == ord('s') and comparing:
                print("Stopping comparison...")
                comparing = False
                if user_sequence:
                    final_score = self._analyze_complete_sequence(user_sequence, reference_landmarks)
                    print(f"Final score: {final_score:.1f}/10")
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
    
    def analyze_sequence_with_dataset(self, landmarks_sequence: List[np.ndarray], exercise_type: str) -> Dict:
        """Analyze sequence against dataset reference"""
        # Load reference
        dataset_file = os.path.join(self.dataset_dir, f"{exercise_type}_perfect.json")
        if not os.path.exists(dataset_file):
            return {"error": f"No dataset found for {exercise_type}"}
        
        with open(dataset_file, 'r') as f:
            dataset_data = json.load(f)
        
        reference_landmarks = [np.array(landmarks) for landmarks in dataset_data['landmarks_sequence']]
        
        # Analyze
        score = self._analyze_complete_sequence(landmarks_sequence, reference_landmarks)
        quality_metrics = self.pose_utils.analyze_movement_quality(landmarks_sequence)
        
        return {
            'exercise_type': exercise_type,
            'dataset_score': score,
            'quality_metrics': quality_metrics,
            'total_frames': len(landmarks_sequence),
            'reference_frames': len(reference_landmarks),
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_real_time_score(self, user_sequence: List[np.ndarray], reference_sequence: List[np.ndarray]) -> float:
        """Calculate real-time similarity score"""
        if not user_sequence or not reference_sequence:
            return 0.0
        
        # Normalize sequences
        user_normalized = self.pose_utils.normalize_pose_sequence(user_sequence)
        ref_normalized = self.pose_utils.normalize_pose_sequence(reference_sequence)
        
        # Calculate similarity (using euclidean distance)
        distance = self.pose_utils.compare_sequences_euclidean(user_normalized, ref_normalized)
        
        # Convert to score (0-10)
        score = max(0, 10 - (distance * 20))
        return min(10, score)
    
    def _analyze_complete_sequence(self, user_sequence: List[np.ndarray], reference_sequence: List[np.ndarray]) -> float:
        """Analyze complete sequence similarity"""
        if not user_sequence or not reference_sequence:
            return 0.0
        
        # Normalize both sequences
        user_normalized = self.pose_utils.normalize_pose_sequence(user_sequence)
        ref_normalized = self.pose_utils.normalize_pose_sequence(reference_sequence)
        
        # Multiple comparison methods
        dtw_distance = self.pose_utils.compare_sequences_dtw(user_normalized, ref_normalized)
        euclidean_distance = self.pose_utils.compare_sequences_euclidean(user_normalized, ref_normalized)
        cosine_distance = self.pose_utils.compare_sequences_cosine(user_normalized, ref_normalized)
        
        # Combine scores
        dtw_score = max(0, 10 - (dtw_distance * 15))
        euclidean_score = max(0, 10 - (euclidean_distance * 20))
        cosine_score = max(0, 10 - (cosine_distance * 30))
        
        # Weighted average
        final_score = (dtw_score * 0.5) + (euclidean_score * 0.3) + (cosine_score * 0.2)
        return min(10, final_score)
    
    def _get_basic_feedback(self, score: float) -> str:
        """Get basic feedback based on score"""
        if score >= 8:
            return "Excellent form!"
        elif score >= 6:
            return "Good form, minor adjustments needed"
        elif score >= 4:
            return "Form needs improvement"
        else:
            return "Focus on proper technique"

# Main functions to be called from main.py
def start_dataset_comparison(exercise_type: str, camera_index: int = 0) -> None:
    """Start real-time dataset comparison"""
    analyzer = DatasetExerciseAnalyzer()
    analyzer.real_time_dataset_comparison(exercise_type, camera_index)

def analyze_with_dataset(landmarks_sequence: List[np.ndarray], exercise_type: str) -> Dict:
    """Analyze sequence with dataset reference"""
    analyzer = DatasetExerciseAnalyzer()
    return analyzer.analyze_sequence_with_dataset(landmarks_sequence, exercise_type)

def list_available_datasets() -> List[str]:
    """List available dataset exercises"""
    return ['squat', 'pushup', 'plank']