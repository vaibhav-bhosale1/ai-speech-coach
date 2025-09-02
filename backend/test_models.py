import cv2
import mediapipe as mp
import sys
import os

# This script will test the MediaPipe Pose model, which is the first one in our chain.

def test_pose_detection(video_path):
    """
    Tests the MediaPipe Pose model on a given video file.
    """
    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found at '{video_path}'")
        return

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open video file '{video_path}'")
        return

    frame_count = 0
    detected_frames = 0
    
    print("\n--- Starting MediaPipe Pose Test ---")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Convert the BGR image to RGB.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image and find pose.
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks:
            detected_frames += 1
            print(f"Frame {frame_count}: ✅ SUCCESS - Pose detected.")
        else:
            print(f"Frame {frame_count}: ❌ FAILURE - No pose detected.")
            
    cap.release()
    pose.close()
    
    print("\n--- Test Complete ---")
    print(f"Total frames processed: {frame_count}")
    print(f"Frames with pose detected: {detected_frames}")
    
    if frame_count > 0 and detected_frames > 0:
        print("\nRESULT: The MediaPipe Pose model appears to be working correctly!")
    else:
        print("\nRESULT: The MediaPipe Pose model FAILED to detect anything. This indicates a problem with the model or your environment.")

if __name__ == '__main__':
    # Use a default video name, which you will create.
    test_video_file = "test_video.webm"
    test_pose_detection(test_video_file)