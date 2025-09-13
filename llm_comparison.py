import json
import numpy as np
import cv2
from typing import List, Dict, Optional
from utils import PoseUtils
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API integration
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI library not available. Install with: pip install openai")

class LLMExerciseAnalyzer:
    def __init__(self):
        self.pose_utils = PoseUtils()
        
        # Get OpenAI API key from environment variable
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if OPENAI_AVAILABLE and self.api_key:
            openai.api_key = self.api_key
        elif OPENAI_AVAILABLE and not self.api_key:
            print("Warning: OPENAI_API_KEY not found in environment variables or .env file")
        
        # Exercise-specific knowledge for LLM prompts
        self.exercise_knowledge = {
            'squat': {
                'description': 'A squat where you lower your body by bending knees and hips, then return to standing',
                'key_points': ['feet shoulder-width apart', 'knees track over toes', 'chest up', 'weight on heels'],
                'common_errors': ['knees caving in', 'forward lean', 'not going deep enough', 'heel lifting']
            },
            'pushup': {
                'description': 'A pushup where you lower chest to ground and push back up while maintaining plank',
                'key_points': ['straight body line', 'hands under shoulders', 'full range of motion', 'controlled movement'],
                'common_errors': ['sagging hips', 'flaring elbows', 'partial range', 'rushing movement']
            },
            'plank': {
                'description': 'A plank hold maintaining straight body position supported on forearms and toes',
                'key_points': ['straight line head to heels', 'engaged core', 'neutral spine', 'steady breathing'],
                'common_errors': ['sagging hips', 'raised hips', 'looking up', 'holding breath']
            }
        }
    
    def real_time_llm_analysis(self, exercise_type: str, camera_index: int = 0) -> None:
        """Real-time exercise analysis with LLM feedback"""
        if not OPENAI_AVAILABLE:
            print("OpenAI library not available. Please install: pip install openai")
            return
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Cannot open camera {camera_index}")
            return
        
        print(f"Starting LLM-powered real-time analysis for {exercise_type}")
        print("Instructions:")
        print("- Position yourself in frame")
        print("- Press 'a' to get AI analysis of your current form")
        print("- Press 'g' to start continuous analysis")
        print("- Press 's' to stop continuous analysis")
        print("- Press 'q' to quit")
        
        user_sequence = []
        analyzing = False
        analysis_interval = 90  # Analyze every 90 frames (about 3 seconds)
        frame_count = 0
        last_analysis_frame = 0
        current_feedback = ["Position yourself and press 'a' for analysis"]
        
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
                
                user_sequence.append(landmarks)
                
                # Continuous analysis
                if analyzing and (frame_count - last_analysis_frame) > analysis_interval:
                    if len(user_sequence) >= 30:  # Need at least 30 frames
                        print("Getting LLM analysis...")
                        feedback = self._get_llm_feedback(user_sequence[-30:], exercise_type)
                        current_feedback = feedback
                        last_analysis_frame = frame_count
                        user_sequence = user_sequence[-60:]  # Keep last 60 frames
            
            # Display feedback on screen
            y_offset = 30
            for i, text in enumerate(current_feedback[:4]):  # Show max 4 lines
                # Split long text
                if len(text) > 50:
                    text = text[:47] + "..."
                cv2.putText(annotated_frame, text, (10, y_offset + i * 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Status
            status = "ANALYZING" if analyzing else "READY"
            cv2.putText(annotated_frame, status, (10, annotated_frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow(f"LLM Analysis - {exercise_type}", annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a') and len(user_sequence) >= 10:
                print("Getting instant LLM analysis...")
                feedback = self._get_llm_feedback(user_sequence[-10:], exercise_type)
                current_feedback = feedback
            elif key == ord('g'):
                print("Starting continuous analysis...")
                analyzing = True
                last_analysis_frame = frame_count
            elif key == ord('s'):
                print("Stopping continuous analysis")
                analyzing = False
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
    
    def analyze_exercise_sequence(self, landmarks_sequence: List[np.ndarray], exercise_type: str) -> Dict:
        """Analyze a complete exercise sequence with LLM"""
        if not OPENAI_AVAILABLE:
            return {"error": "OpenAI library not available"}
        
        if not landmarks_sequence:
            return {"error": "No landmarks provided"}
        
        print(f"Analyzing {len(landmarks_sequence)} frames with LLM...")
        
        # Get movement analysis data
        movement_data = self._analyze_movement_patterns(landmarks_sequence)
        
        # Get LLM feedback
        feedback = self._get_detailed_llm_analysis(movement_data, exercise_type)
        
        return {
            'exercise_type': exercise_type,
            'total_frames': len(landmarks_sequence),
            'movement_analysis': movement_data,
            'llm_feedback': feedback,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_llm_feedback(self, landmarks_sequence: List[np.ndarray], exercise_type: str) -> List[str]:
        """Get quick LLM feedback for real-time use"""
        if not landmarks_sequence:
            return ["No pose detected"]
        
        # Analyze current movement
        movement_data = self._analyze_movement_patterns(landmarks_sequence)
        
        # Create prompt for LLM
        prompt = self._create_quick_analysis_prompt(movement_data, exercise_type)
        
        try:
            # Get LLM response
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a fitness coach. Give brief, actionable feedback."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.3
            )
            
            feedback_text = response.choices[0].message.content.strip()
            
            # Split into lines for display
            return self._format_feedback_for_display(feedback_text)
            
        except Exception as e:
            print(f"LLM API error: {e}")
            return ["LLM analysis unavailable", "Check your form manually"]
    
    def _get_detailed_llm_analysis(self, movement_data: Dict, exercise_type: str) -> Dict:
        """Get detailed LLM analysis for complete sequences"""
        prompt = self._create_detailed_analysis_prompt(movement_data, exercise_type)
        
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert fitness coach and biomechanics analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            feedback_text = response.choices[0].message.content.strip()
            
            return {
                'raw_feedback': feedback_text,
                'parsed_sections': self._parse_detailed_feedback(feedback_text)
            }
            
        except Exception as e:
            print(f"LLM API error: {e}")
            return {"error": str(e)}
    
    def _analyze_movement_patterns(self, landmarks_sequence: List[np.ndarray]) -> Dict:
        """Analyze movement patterns from landmarks"""
        if not landmarks_sequence:
            return {}
        
        # Calculate basic metrics
        quality_metrics = self.pose_utils.analyze_movement_quality(landmarks_sequence)
        
        # Calculate joint angles over time
        angle_data = []
        for landmarks in landmarks_sequence:
            angles = self.pose_utils.calculate_joint_angles_from_landmarks(landmarks)
            angle_data.append(angles)
        
        # Analyze key joint movements
        movement_analysis = {}
        
        if angle_data:
            # Get average and range for key joints
            key_joints = ['left_knee', 'right_knee', 'left_elbow', 'right_elbow', 'left_hip', 'right_hip']
            
            for joint in key_joints:
                joint_angles = [frame.get(joint, 0) for frame in angle_data if joint in frame]
                if joint_angles:
                    movement_analysis[joint] = {
                        'average': float(np.mean(joint_angles)),
                        'range': float(np.max(joint_angles) - np.min(joint_angles)),
                        'min': float(np.min(joint_angles)),
                        'max': float(np.max(joint_angles))
                    }
        
        return {
            'quality_metrics': quality_metrics,
            'joint_analysis': movement_analysis,
            'total_frames': len(landmarks_sequence)
        }
    
    def _create_quick_analysis_prompt(self, movement_data: Dict, exercise_type: str) -> str:
        """Create prompt for quick real-time feedback"""
        exercise_info = self.exercise_knowledge.get(exercise_type, {})
        
        prompt = f"""
Analyze this {exercise_type} exercise performance and give 2-3 brief coaching tips:

Exercise: {exercise_info.get('description', exercise_type)}
Key points to check: {', '.join(exercise_info.get('key_points', []))}

Movement Quality:
- Smoothness: {movement_data.get('quality_metrics', {}).get('smoothness', 0):.2f}
- Consistency: {movement_data.get('quality_metrics', {}).get('consistency', 0):.2f}

Joint Analysis:
"""
        
        joint_analysis = movement_data.get('joint_analysis', {})
        for joint, data in joint_analysis.items():
            if 'average' in data:
                prompt += f"- {joint}: avg {data['average']:.1f}°, range {data['range']:.1f}°\n"
        
        prompt += f"""
Common errors to watch for: {', '.join(exercise_info.get('common_errors', []))}

Give 2-3 brief, actionable coaching tips (max 50 characters each):
"""
        
        return prompt
    
    def _create_detailed_analysis_prompt(self, movement_data: Dict, exercise_type: str) -> str:
        """Create prompt for detailed analysis"""
        exercise_info = self.exercise_knowledge.get(exercise_type, {})
        
        prompt = f"""
Provide a comprehensive analysis of this {exercise_type} exercise performance:

EXERCISE DETAILS:
- Type: {exercise_info.get('description', exercise_type)}
- Key technique points: {', '.join(exercise_info.get('key_points', []))}
- Common errors: {', '.join(exercise_info.get('common_errors', []))}

PERFORMANCE DATA:
- Total frames analyzed: {movement_data.get('total_frames', 0)}

Movement Quality Metrics:
- Smoothness: {movement_data.get('quality_metrics', {}).get('smoothness', 0):.3f} (0-1 scale)
- Consistency: {movement_data.get('quality_metrics', {}).get('consistency', 0):.3f} (0-1 scale)
- Amplitude: {movement_data.get('quality_metrics', {}).get('amplitude', 0):.3f}

Joint Angle Analysis:
"""
        
        joint_analysis = movement_data.get('joint_analysis', {})
        for joint, data in joint_analysis.items():
            prompt += f"- {joint}: average {data.get('average', 0):.1f}°, range {data.get('range', 0):.1f}° (min: {data.get('min', 0):.1f}°, max: {data.get('max', 0):.1f}°)\n"
        
        prompt += """
Please provide:
1. OVERALL ASSESSMENT (1-2 sentences)
2. SPECIFIC TECHNIQUE FEEDBACK (3-4 key points)  
3. SAFETY CONSIDERATIONS (if any)
4. IMPROVEMENT SUGGESTIONS (2-3 actionable items)

Focus on the most important aspects for safe and effective exercise performance.
"""
        
        return prompt
    
    def _format_feedback_for_display(self, feedback_text: str) -> List[str]:
        """Format LLM feedback for real-time display"""
        # Split by sentences or lines
        lines = []
        
        # Split by periods or newlines
        sentences = feedback_text.replace('\n', '. ').split('. ')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 5:
                # Break long sentences into multiple lines
                if len(sentence) > 50:
                    words = sentence.split()
                    current_line = ""
                    for word in words:
                        if len(current_line + word) > 50:
                            if current_line:
                                lines.append(current_line.strip())
                                current_line = word + " "
                            else:
                                lines.append(word)
                                current_line = ""
                        else:
                            current_line += word + " "
                    if current_line.strip():
                        lines.append(current_line.strip())
                else:
                    lines.append(sentence)
        
        return lines[:4]  # Return max 4 lines
    
    def _parse_detailed_feedback(self, feedback_text: str) -> Dict:
        """Parse detailed feedback into sections"""
        sections = {
            'overall_assessment': '',
            'technique_feedback': [],
            'safety_considerations': [],
            'improvement_suggestions': []
        }
        
        current_section = None
        lines = feedback_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect section headers
            if 'OVERALL ASSESSMENT' in line.upper():
                current_section = 'overall_assessment'
            elif 'TECHNIQUE FEEDBACK' in line.upper():
                current_section = 'technique_feedback'
            elif 'SAFETY' in line.upper():
                current_section = 'safety_considerations'
            elif 'IMPROVEMENT' in line.upper():
                current_section = 'improvement_suggestions'
            elif current_section and line.startswith(('•', '-', '1.', '2.', '3.', '4.')):
                # Bullet point or numbered item
                content = line.lstrip('•-1234. ').strip()
                if current_section in ['technique_feedback', 'safety_considerations', 'improvement_suggestions']:
                    sections[current_section].append(content)
            elif current_section == 'overall_assessment' and not line.startswith(('•', '-')):
                # Add to overall assessment
                if sections['overall_assessment']:
                    sections['overall_assessment'] += ' ' + line
                else:
                    sections['overall_assessment'] = line
        
        return sections

# Main functions to be called from main.py
def start_llm_analysis(exercise_type: str, camera_index: int = 0) -> None:
    """Start real-time LLM-powered exercise analysis"""
    analyzer = LLMExerciseAnalyzer()
    analyzer.real_time_llm_analysis(exercise_type, camera_index)

def analyze_with_llm(landmarks_sequence: List[np.ndarray], exercise_type: str) -> Dict:
    """Analyze exercise sequence with LLM"""
    analyzer = LLMExerciseAnalyzer()
    return analyzer.analyze_exercise_sequence(landmarks_sequence, exercise_type)

def test_llm_connection() -> bool:
    """Test if LLM connection is working"""
    if not OPENAI_AVAILABLE:
        print("OpenAI library not installed")
        return False
    
    try:
        analyzer = LLMExerciseAnalyzer()
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, are you working?"}],
            max_tokens=10
        )
        print("LLM connection successful!")
        return True
    except Exception as e:
        print(f"LLM connection failed: {e}")
        return False