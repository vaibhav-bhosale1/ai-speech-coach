import os
import re
import numpy as np
import librosa
import cv2
import subprocess
import mediapipe as mp
from collections import Counter
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deepface import DeepFace

# --- Constants & Setup ---
FILLER_WORDS = [
    'um', 'uh', 'er', 'ah', 'like', 'okay', 'right', 'so', 'you know', 
    'basically', 'actually', 'literally', 'well', 'i mean'
]
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AI Model Loading ---
try:
    whisper_model = WhisperModel("small.en", device="cpu", compute_type="int8")
    sentiment_analyzer = SentimentIntensityAnalyzer()
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    print("✅ AI models (Whisper, VADER, MediaPipe, Pose) loaded successfully.")
except Exception as e:
    print(f"🔥 Failed to load an AI model: {e}")
    whisper_model = sentiment_analyzer = face_mesh = pose = None

# --- Analysis Helper Functions (Audio) ---
def calculate_wpm(transcript, duration_seconds):
    word_count = len(transcript.split())
    return round((word_count / duration_seconds) * 60) if duration_seconds > 0 else 0

def count_filler_words(transcript):
    words = re.findall(r'\b\w+\b', transcript.lower())
    counts = Counter(word for word in words if word in FILLER_WORDS)
    return dict(counts)

def analyze_audio_quality(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        if np.max(np.abs(y)) < 0.005:
            return {"pitch_variation": "N/A", "volume_consistency": "N/A", "confidence_score": "N/A"}
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        non_zero_pitches = pitches[pitches > 0]
        pitch_std_dev = np.std(non_zero_pitches) if len(non_zero_pitches) > 0 else 0.0
        pitch_variation = "Dynamic" if pitch_std_dev > 25 else "Monotone"
        rms = librosa.feature.rms(y=y)[0]
        rms_std_dev_db = np.std(librosa.amplitude_to_db(rms)) if len(rms) > 0 else 50.0
        volume_consistency = max(0, 100 - rms_std_dev_db * 5)
        confidence_score = (volume_consistency + (90 if pitch_variation == "Dynamic" else 50)) / 2
        return {"pitch_variation": pitch_variation, "volume_consistency": round(volume_consistency), "confidence_score": round(confidence_score)}
    except Exception as e:
        print(f"🔥 Audio quality analysis error: {e}")
        return {"pitch_variation": "Error", "volume_consistency": 0, "confidence_score": 0}

def analyze_sentiment(transcript):
    if not sentiment_analyzer or not transcript: return {"label": "N/A"}
    scores = sentiment_analyzer.polarity_scores(transcript)
    compound = scores['compound']
    label = "Positive" if compound > 0.05 else "Negative" if compound < -0.05 else "Neutral"
    return {"label": label, "score": scores}

# --- FINAL, FOCUSED Analysis Helper Function (Video) ---
def analyze_video_features(video_path):
    if not face_mesh or not pose:
        return {
            "video_analysis": {"dominant_emotion": "Error", "emotion_distribution": {}},
            "eye_contact": {"gaze_stability": "Error"},
            "posture": {"score": 0, "tilt_status": "Error"}
        }

    cap = cv2.VideoCapture(video_path)
    emotions = []
    forward_gaze_frames = 0
    posture_scores = []
    head_tilt_warnings = 0
    
    total_frames = 0
    analysis_frames_count = 0 
    frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30
    
    LEFT_IRIS = [474, 475, 476, 477]
    LEFT_EYE_CORNERS = [33, 133]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        total_frames += 1

        # Run all heavy analysis only once per second for performance
        if total_frames % int(frame_rate) == 0:
            analysis_frames_count += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --- 1. Posture Analysis (MediaPipe) ---
            pose_results = pose.process(frame_rgb)
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                nose = landmarks[mp_pose.PoseLandmark.NOSE]

                if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
                    shoulder_y_diff = abs(left_shoulder.y - right_shoulder.y)
                    score = max(0, 100 - (shoulder_y_diff * 1000))
                    posture_scores.append(score)
                if nose.visibility > 0.5 and left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
                    shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
                    head_tilt_diff = abs(nose.x - shoulder_center_x)
                    if head_tilt_diff > 0.08:
                        head_tilt_warnings += 1

            # --- 2. Eye Contact Analysis (MediaPipe) ---
            face_mesh_results = face_mesh.process(frame_rgb)
            if face_mesh_results.multi_face_landmarks:
                face_landmarks = face_mesh_results.multi_face_landmarks[0].landmark
                l_iris_center_norm = np.mean([(face_landmarks[i].x, face_landmarks[i].y) for i in LEFT_IRIS], axis=0)
                l_corner_left_norm = np.array([face_landmarks[LEFT_EYE_CORNERS[0]].x, face_landmarks[LEFT_EYE_CORNERS[0]].y])
                l_corner_right_norm = np.array([face_landmarks[LEFT_EYE_CORNERS[1]].x, face_landmarks[LEFT_EYE_CORNERS[1]].y])
                eye_width_l = np.linalg.norm(l_corner_right_norm - l_corner_left_norm)
                iris_to_corner_l = np.linalg.norm(l_iris_center_norm - l_corner_left_norm)
                ratio_l = iris_to_corner_l / eye_width_l if eye_width_l > 0 else 0.5
                if 2.70 < ratio_l < 3.10:
                    forward_gaze_frames += 1
            
            # --- 3. Emotion Analysis (DeepFace - using powerful detector) ---
            try:
                analysis = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False, detector_backend='mtcnn')
                if analysis and len(analysis) > 0:
                    emotions.append(analysis[0]['dominant_emotion'].capitalize())
            except Exception as e:
                print(f"DEBUG: DeepFace analysis failed. Error: {e}")

    cap.release()

    # --- Process all collected data ---
    if emotions:
        emotion_counts = Counter(emotions)
        dominant_emotion_final = emotion_counts.most_common(1)[0][0]
        distribution = {k: round(v / len(emotions) * 100) for k, v in emotion_counts.items()}
        video_analysis_result = {"dominant_emotion": dominant_emotion_final, "emotion_distribution": distribution}
    else:
        video_analysis_result = {"dominant_emotion": "N/A", "emotion_distribution": {}}

    gaze_stability = round((forward_gaze_frames / analysis_frames_count) * 100) if analysis_frames_count > 0 else 0
    eye_contact_result = {"gaze_stability": gaze_stability}

    avg_posture_score = round(np.mean(posture_scores)) if posture_scores else 0
    tilt_percentage = (head_tilt_warnings / analysis_frames_count) * 100 if analysis_frames_count > 0 else 0
    tilt_status = "Good"
    if tilt_percentage > 30:
        tilt_status = "Needs Improvement"
    posture_result = {"score": avg_posture_score, "tilt_status": tilt_status}

    return {
        "video_analysis": video_analysis_result, 
        "eye_contact": eye_contact_result,
        "posture": posture_result
    }

# --- API Endpoint ---
@app.post("/analyze")
async def analyze_performance(media_file: UploadFile = File(...)):
    if not all([whisper_model, sentiment_analyzer, face_mesh, pose]):
        raise HTTPException(500, "AI models are not available.")

    temp_video_path = f"temp_{media_file.filename}"
    temp_audio_path = "temp_extracted_audio.wav"
    
    try:
        with open(temp_video_path, "wb") as f: f.write(await media_file.read())
        
        subprocess.run(
            ['ffmpeg', '-i', temp_video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', temp_audio_path, '-y'], 
            check=True, capture_output=True
        )

        segments, _ = whisper_model.transcribe(
            temp_audio_path, beam_size=5, vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        transcript = "".join(segment.text for segment in segments).strip()
        duration = librosa.get_duration(path=temp_audio_path)
        
        video_features = analyze_video_features(temp_video_path)

        if not transcript:
            return {
                "transcription": "No speech detected.", "duration": round(duration, 2),
                "wpm": 0, "filler_words": {}, "sentiment": {"label": "N/A"},
                "audio_quality": analyze_audio_quality(temp_audio_path),
                "video_analysis": video_features["video_analysis"],
                "eye_contact": video_features["eye_contact"],
                "posture": video_features["posture"]
            }
        
        return {
            "transcription": transcript, "duration": round(duration, 2),
            "wpm": calculate_wpm(transcript, duration),
            "filler_words": count_filler_words(transcript),
            "sentiment": analyze_sentiment(transcript),
            "audio_quality": analyze_audio_quality(temp_audio_path),
            "video_analysis": video_features["video_analysis"],
            "eye_contact": video_features["eye_contact"],
            "posture": video_features["posture"]
        }
    except subprocess.CalledProcessError as e:
        raise HTTPException(500, f"FFmpeg Error: {e.stderr.decode()}")
    except Exception as e:
        print(f"🔥 Unhandled analysis error: {e}")
        raise HTTPException(500, f"An unexpected error occurred: {e}")
    finally:
        if os.path.exists(temp_video_path): os.remove(temp_video_path)
        if os.path.exists(temp_audio_path): os.remove(temp_audio_path)