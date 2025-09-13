#!/usr/bin/env python3
"""
Simple Main File - Complete Pose Analysis System
Just calls functions from the three comparison modules
"""

from coach_comparison import (
    process_coach_video, 
    start_coach_comparison, 
    list_available_exercises
)
from llm_comparison import (
    start_llm_analysis,
    test_llm_connection
)
from dataset_comparison import (
    start_dataset_comparison,
    list_available_datasets
)

def main():
    print("="*60)
    print("COMPLETE POSE ANALYSIS SYSTEM")
    print("="*60)
    
    while True:
        print("\nChoose an option:")
        print("1. Coach Video Comparison")
        print("2. LLM-based Analysis") 
        print("3. Dataset Comparison")
        print("4. Test LLM Connection")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            handle_coach_comparison()
        elif choice == '2':
            handle_llm_analysis()
        elif choice == '3':
            handle_dataset_comparison()
        elif choice == '4':
            test_llm_connection()
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

def handle_coach_comparison():
    print("\n--- COACH VIDEO COMPARISON ---")
    
    # Show available processed exercises
    processed_exercises = list_available_exercises()
    if processed_exercises:
        print("Processed exercises available for comparison:", processed_exercises)
    else:
        print("No processed exercises found. You need to process a coach video first.")
    
    print("\nOptions:")
    print("1. Process new coach video")
    print("2. Start real-time comparison")
    print("3. Back to main menu")
    
    choice = input("Choose option (1-3): ").strip()
    
    if choice == '1':
        # Process new coach video
        exercise_type = input("Enter exercise type: ").strip().lower()
        video_path = input("Enter path to coach video file: ").strip()
        
        # Remove quotes if present
        if video_path.startswith('"') and video_path.endswith('"'):
            video_path = video_path[1:-1]
        if video_path.startswith("'") and video_path.endswith("'"):
            video_path = video_path[1:-1]
        
        print(f"Processing coach video for {exercise_type} from: {video_path}")
        success = process_coach_video(video_path, exercise_type)
        
        if success:
            print("Coach video processed successfully!")
            print(f"You can now use real-time comparison for '{exercise_type}'")
        else:
            print("Failed to process coach video. Please check the video path and try again.")
    
    elif choice == '2':
        # Start real-time comparison
        if not processed_exercises:
            print("No processed exercises available. Process a coach video first.")
            return
            
        print("Available exercises:", processed_exercises)
        exercise = input("Enter exercise type: ").strip().lower()
        
        if exercise not in processed_exercises:
            print(f"Exercise '{exercise}' not found. Available: {processed_exercises}")
            return
        
        camera = input("Enter camera index (default 0): ").strip()
        camera_index = int(camera) if camera.isdigit() else 0
        
        print(f"Starting coach comparison for {exercise}...")
        start_coach_comparison(exercise, camera_index)
    
    elif choice == '3':
        return
    else:
        print("Invalid choice.")

def handle_llm_analysis():
    print("\n--- LLM-BASED ANALYSIS ---")
    
    # Test connection first
    print("Testing LLM connection...")
    if not test_llm_connection():
        print("LLM connection failed. Please check your API key.")
        return
    
    exercise = input("Enter exercise type (squat/pushup/plank): ").strip().lower()
    camera = input("Enter camera index (default 0): ").strip()
    camera_index = int(camera) if camera.isdigit() else 0
    
    print(f"Starting LLM analysis for {exercise}...")
    start_llm_analysis(exercise, camera_index)

def handle_dataset_comparison():
    print("\n--- DATASET COMPARISON ---")
    print("Available datasets:", list_available_datasets())
    
    exercise = input("Enter exercise type (squat/pushup/plank): ").strip().lower()
    camera = input("Enter camera index (default 0): ").strip()
    camera_index = int(camera) if camera.isdigit() else 0
    
    if exercise in list_available_datasets():
        print(f"Starting dataset comparison for {exercise}...")
        start_dataset_comparison(exercise, camera_index)
    else:
        print("Invalid exercise type.")


main()