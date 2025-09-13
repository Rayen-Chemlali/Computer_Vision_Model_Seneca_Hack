# Complete Pose Analysis System

A comprehensive exercise form analysis system that uses MediaPipe pose detection and three comparison methods: Coach Video Comparison, Dataset Comparison, and LLM-based Evaluation.

## 🎯 Features

- **Real-time pose detection** using MediaPipe (33 body landmarks)
- **Coach Video Comparison** - Compare against reference coach videos
- **Dataset Comparison** - Compare against standardized exercise datasets
- **LLM-based Evaluation** - Get human-like feedback using GPT-4/5
- **Multi-modal analysis** - Joint angles, movement patterns, quality metrics
- **Real-time and file-based processing**
- **Comprehensive reporting** - Detailed analysis reports

## 📁 File Structure

```
pose_analysis_system/
├── utils.py                 # Shared utilities and pose processing
├── coach_comparison.py       # Coach video comparison module
├── dataset_comparison.py     # Dataset-based comparison module
├── llm_comparison.py        # LLM-powered evaluation module
├── main.py                  # Main integration system
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── coach_data/             # Processed coach video data
├── datasets/               # Exercise datasets
└── reports/                # Generated analysis reports
```

## 🛠 Installation

### Prerequisites

```bash
# Install Python dependencies
pip install opencv-python mediapipe numpy pandas matplotlib
pip install scikit-learn scipy dtaidistance requests
pip install openai  # For LLM integration
```

### MediaPipe Requirements
```bash
# MediaPipe should install automatically, but if issues occur:
pip install mediapipe==0.10.9
```

### Optional Dependencies
```bash
# For enhanced dataset processing
pip install kaggle  # If using Kaggle datasets
pip install yt-dlp  # If processing YouTube videos
```

## 🚀 Quick Start

### 1. Basic Video Analysis

```python
from main import CompletePoseAnalysisSystem

# Initialize system
system = CompletePoseAnalysisSystem()

# Analyze a user video
results = system.analyze_user_video(
    video_path="user_squat.mp4",
    exercise_name="squat",
    comparison_methods=['dataset', 'llm']
)

print(f"Overall performance: {results['comprehensive_summary']['overall_performance']}")
```

### 2. Real-time Analysis

```python
# Real-time webcam analysis
system.real_time_analysis(
    exercise_name="pushup",
    comparison_methods=['dataset'],
    camera_index=0
)
```

### 3. Command Line Usage

```bash
# Analyze video file
python main.py --video user_exercise.mp4 --exercise squat --methods dataset llm

# Real-time analysis
python main.py --exercise pushup --realtime --methods dataset

# Setup coach videos
python main.py --exercise squat --setup-coaches coach_squat1.mp4 coach_squat2.mp4
```

## 📊 Dataset Integration

### Evaluating Your Datasets

The system can evaluate your provided datasets and determine their usefulness:

```python
# Evaluate your datasets
evaluation = system.evaluate_datasets([
    "fitness-activities.txt",
    "fitness-nutrition.txt", 
    "fitness-users.txt",
    "sleep_data_sample.json"
])
```

**Current Assessment of Your Datasets:**
- `fitness-activities.txt`: Contains fitness metrics but lacks pose landmark data
- `fitness-nutrition.txt`: Nutrition data, not suitable for pose comparison
- `fitness-users.txt`: User profiles, not movement data
- `sleep_data_sample.json`: Sleep tracking data, unrelated to exercise form

**Recommendation:** The system will use high-quality alternative datasets:

### Alternative Datasets (Automatically Used)

1. **AI Fitness Trainer Dataset**
   - Source: https://github.com/Dene33/ai-fitness-trainer
   - Contains: Pushups, squats, pullups with pose landmarks
   - Quality: High

2. **Human3.6M Dataset**
   - Source: http://vision.imar.ro/human3.6m/
   - Contains: 3D human poses for various activities
   - Quality: High (research-grade)

3. **Synthetic Perfect Form Dataset** (Generated)
   - Contains: Perfect form references for squat, pushup, lunge, deadlift
   - Quality: High (mathematically perfect)

4. **Yoga Pose Dataset**
   - Source: https://www.kaggle.com/datasets/tr1gg3rtrash/yoga-posture-dataset
   - Contains: Yoga poses with detailed landmarks
   - Quality: High

## 🎥 Coach Video Setup

### Processing Coach Videos

```python
# Process a coach reference video
system.coach_comparison.process_coach_video(
    video_path="coach_perfect_squat.mp4",
    exercise_name="squat",
    coach_name="john_smith"
)

# Use in analysis
results = system.analyze_user_video(
    "user_squat.mp4", 
    "squat", 
    methods=['coach'],
    coach_name="john_smith"
)
```

### Coach Video Requirements
- Clear visibility of the person
- Good lighting
- Full body in frame
- Demonstrates proper form
- 30-120 seconds duration

## 🤖 LLM Integration

### OpenAI Setup

```python
import os

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = 'your-api-key-here'

# Or pass directly
system = CompletePoseAnalysisSystem(llm_api_key='your-api-key')
```

### LLM Analysis Features
- Context-aware feedback
- Exercise-specific guidance  
- Natural language explanations
- Safety considerations
- Progressive recommendations

### Local LLM Support

```python
from llm_comparison import LLMConfig

# For local LLMs (Ollama, LocalAI, etc.)
llm_config = LLMConfig(
    provider="local",
    base_url="http://localhost:11434",  # Ollama default
    model="llama3:8b"
)
```

## 📈 Analysis Methods

### 1. Coach Video Comparison
- **DTW (Dynamic Time Warping)** - Temporal alignment
- **Euclidean Distance** - Spatial similarity  
- **Joint Angle Analysis** - Biomechanical comparison
- **Key Pose Matching** - Critical position analysis

### 2. Dataset Comparison
- **Statistical Analysis** - Movement pattern statistics
- **Quality Metrics** - Smoothness, consistency, amplitude
- **Reference Matching** - Best dataset selection
- **Confidence Scoring** - Reliability assessment

### 3. LLM Evaluation
- **Contextual Understanding** - Exercise-specific knowledge
- **Natural Feedback** - Human-like explanations
- **Safety Assessment** - Injury risk evaluation
- **Progressive Guidance** - Skill development suggestions

## 📋 Output Reports

### Comprehensive Analysis Report

```python
# Generate detailed report
results = system.analyze_user_video("exercise.mp4", "squat")

# Access different aspects
print("Overall Score:", results['comprehensive_summary']['overall_performance'])
print("Key Insights:", results['comprehensive_summary']['key_insights'])
print("Improvements:", results['comprehensive_summary']['priority_improvements'])
```

### Report Contents
- **Overall Performance Rating**
- **Method-specific Scores**
- **Technical Metrics** (angles, smoothness, etc.)
- **Actionable Feedback**
- **Positive Reinforcement**
- **Progress Recommendations**

## ⚙️ Configuration Options

### System Configuration

```python
# Custom configuration
system = CompletePoseAnalysisSystem()

# MediaPipe settings (modify in utils.py)
pose = mp.solutions.pose.Pose(
    static_image_mode=False,
    model_complexity=2,  # 0=lite, 1=full, 2=heavy
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
```

### Analysis Parameters

```python
# Video analysis options
results = system.analyze_user_video(
    video_path="video.mp4",
    exercise_name="squat",
    comparison_methods=['coach', 'dataset', 'llm'],
    coach_name='expert_trainer',
    save_reports=True  # Generate detailed reports
)

# Real-time options
system.real_time_analysis(
    exercise_name="pushup",
    comparison_methods=['dataset'],
    camera_index=0,
    analysis_interval=60  # Analyze every 60 frames
)
```

## 🎯 Supported Exercises

### Primary Support
- **Squats** - Full depth analysis, knee tracking
- **Push-ups** - Elbow angles, body alignment
- **Lunges** - Balance, leg positioning
- **Deadlifts** - Hip hinge, back position

### Extended Support  
- **Planks** - Core stability
- **Yoga Poses** - Flexibility assessment
- **General Movements** - Basic pose analysis

## 🔧 Troubleshooting

### Common Issues

#### MediaPipe Not Detecting Poses
```python
# Check video quality
# Ensure good lighting
# Person should be fully visible
# Try adjusting detection confidence
pose = mp.solutions.pose.Pose(min_detection_confidence=0.5)
```

#### LLM API Errors
```python
# Check API key
import os
print(os.environ.get('OPENAI_API_KEY'))

# Check rate limits
# Reduce analysis frequency for real-time mode
```

#### Dataset Loading Issues
```python
# Check dataset file paths
# Ensure proper file formats (JSON, CSV)
# Verify pose landmark data structure
```

### Performance Optimization

```python
# For real-time performance
system.real_time_analysis(
    exercise_name="squat",
    comparison_methods=['dataset'],  # Skip LLM for speed
    analysis_interval=120  # Analyze less frequently
)

# For video processing
results = system.analyze_user_video(
    "video.mp4", "squat",
    comparison_methods=['dataset'],  # Fastest method
    save_reports=False  # Skip detailed reports
)
```

## 📚 Advanced Usage

### Batch Processing

```python
# Process multiple videos
video_configs = [
    {'video_path': 'user1_squat.mp4', 'exercise_name': 'squat'},
    {'video_path': 'user2_pushup.mp4', 'exercise_name': 'pushup'},
    {'video_path': 'user3_lunge.mp4', 'exercise_name': 'lunge'}
]

batch_results = system.batch_analyze_videos(video_configs)
```

### Custom Dataset Integration

```python
# Add custom dataset
system.dataset_comparison.datasets['my_custom_squats'] = {
    'exercise_name': 'squat',
    'landmarks_sequence': custom_landmarks_data,
    'description': 'My custom perfect squat dataset'
}
```

### Integration with Existing Systems

```python
# Extract just the pose landmarks
landmarks = system.pose_utils.extract_landmarks_from_video("video.mp4")

# Use individual comparison methods
coach_result = system.coach_comparison.compare_with_coach(landmarks, "squat")
dataset_result = system.dataset_comparison.compare_with_dataset(landmarks, "squat") 
llm_result = system.llm_comparison.compare_with_llm(landmarks, exercise_name="squat")
```

## 🚨 Safety Considerations

The system is designed to promote safe exercise practices:

- **Form Analysis**: Identifies potentially harmful movement patterns
- **Range of Motion**: Checks for excessive or insufficient movement
- **Joint Alignment**: Monitors for injury-prone positions
- **Progressive Feedback**: Encourages gradual improvement

**Note**: This system provides guidance for exercise form improvement but is not a substitute for professional training or medical advice.

## 📄 License

This project is provided for educational and research purposes. Please ensure you have appropriate permissions for any datasets or videos you use with the system.

## 🤝 Contributing

The system is modular and extensible. You can:
- Add new exercise types
- Integrate additional datasets
- Extend the LLM prompts
- Add new comparison methods

## 📞 Support

For technical issues:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure proper file formats and permissions
4. Check API keys and network connectivity

---

**Ready to analyze your exercise form!** Start with the quick examples above and customize the system for your specific needs.
