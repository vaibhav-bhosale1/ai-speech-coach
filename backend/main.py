import os
import re
import numpy as np
import librosa
import cv2
import subprocess
import mediapipe as mp
import sqlalchemy
from collections import Counter
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
from supabase import create_client, Client

# --- Environment & Setup ---
load_dotenv()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- Supabase & Database Setup ---
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    engine = sqlalchemy.create_engine(DATABASE_URL)
    print("✅ Successfully connected to Supabase.")
except Exception as e:
    print(f"🔥 Failed to connect to Supabase: {e}")
    supabase = engine = None

# --- AI Model Loading (from your code) ---
try:
    whisper_model = WhisperModel("small.en", device="cpu", compute_type="int8")
    sentiment_analyzer = SentimentIntensityAnalyzer()
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.3)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3)
    
    # Ensure you have these model files in a 'models' directory
    face_proto = "models/deploy.prototxt"
    face_model = "models/res10_300x300_ssd_iter_140000.caffemodel"
    emotion_model_path = "models/emotion-ferplus-8.onnx"
    
    face_net = cv2.dnn.readNet(face_proto, face_model)
    emotion_net = cv2.dnn.readNetFromONNX(emotion_model_path)
    EMOTION_LABELS = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Anger', 'Disgust', 'Fear', 'Contempt']
    print("✅ AI models loaded successfully.")
except Exception as e:
    print(f"🔥 Failed to load an AI model: {e}")
    whisper_model = sentiment_analyzer = face_mesh = pose = face_net = emotion_net = None

# --- Authentication Dependency ---
def get_user(request: Request):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not token: raise HTTPException(status_code=401, detail="Auth token missing")
    
    # FIX: This line tells the Supabase client to act on behalf of the user
    # for all subsequent database operations in this request.
    supabase.postgrest.auth(token)

    try:
        user = supabase.auth.get_user(token).user
        if not user: raise HTTPException(status_code=401, detail="Invalid token")
        return user
    except Exception: raise HTTPException(status_code=401, detail="Could not validate credentials")

# --- Analysis Helper Functions (from your code) ---
FILLER_WORDS = ['um', 'uh', 'er', 'ah', 'like', 'okay', 'right', 'so', 'you know', 'basically', 'actually', 'literally', 'well', 'i mean']
def calculate_wpm(transcript, duration_seconds):
    word_count = len(transcript.split())
    return round((word_count / duration_seconds) * 60) if duration_seconds > 0 else 0
def count_filler_words(transcript):
    words = re.findall(r'\b\w+\b', transcript.lower())
    return dict(Counter(word for word in words if word in FILLER_WORDS))
def analyze_audio_quality(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        if np.max(np.abs(y)) < 0.005: return {"pitch_variation": "N/A", "volume_consistency": "N/A", "confidence_score": "N/A"}
        pitches, _ = librosa.piptrack(y=y, sr=sr); non_zero_pitches = pitches[pitches > 0]
        pitch_std_dev = np.std(non_zero_pitches) if len(non_zero_pitches) > 0 else 0.0
        pitch_variation = "Dynamic" if pitch_std_dev > 25 else "Monotone"
        rms = librosa.feature.rms(y=y)[0]
        rms_std_dev_db = np.std(librosa.amplitude_to_db(rms)) if len(rms) > 0 else 50.0
        volume_consistency = max(0, 100 - rms_std_dev_db * 5)
        confidence_score = (volume_consistency + (90 if pitch_variation == "Dynamic" else 50)) / 2
        return {"pitch_variation": pitch_variation, "volume_consistency": round(volume_consistency), "confidence_score": round(confidence_score)}
    except Exception: return {"pitch_variation": "Error", "volume_consistency": 0, "confidence_score": 0}
def analyze_sentiment(transcript):
    if not sentiment_analyzer or not transcript: return {"label": "N/A"}
    scores = sentiment_analyzer.polarity_scores(transcript); compound = scores['compound']
    label = "Positive" if compound > 0.05 else "Negative" if compound < -0.05 else "Neutral"
    return {"label": label, "score": scores}
def analyze_video_features(video_path):
    # Your full, detailed analyze_video_features function
    if not all([face_mesh, pose, face_net, emotion_net]): return {"video_analysis": {"dominant_emotion": "Error"}, "eye_contact": {"gaze_stability": "Error"}, "posture": {"score": 0}}
    cap = cv2.VideoCapture(video_path); emotions = []; forward_gaze_frames = 0; posture_scores = []; head_tilt_warnings = 0; total_frames = 0;
    LEFT_IRIS = [474, 475, 476, 477]; LEFT_EYE_CORNERS = [33, 133]
    try:
        while cap.isOpened():
            ret, frame = cap.read();
            if not ret: break
            total_frames += 1; frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            (h, w) = frame.shape[:2]; blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            face_net.setInput(blob); detections = face_net.forward()
            if detections.shape[2] > 0 and detections[0, 0, 0, 2] > 0.3:
                box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h]); (startX, startY, endX, endY) = box.astype("int")
                face_roi = frame[startY:endY, startX:endX]; gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                resized_face = cv2.resize(gray_face, (64, 64)); emotion_blob = resized_face.astype("float32").reshape(1, 1, 64, 64)
                emotion_net.setInput(emotion_blob); preds = emotion_net.forward(); emotions.append(EMOTION_LABELS[preds.argmax()].capitalize())
            pose_results = pose.process(frame_rgb)
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark; left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]; right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]; nose = landmarks[mp_pose.PoseLandmark.NOSE]
                if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5: posture_scores.append(max(0, 100 - (abs(left_shoulder.y - right_shoulder.y) * 1000)))
                if nose.visibility > 0.5 and left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5 and abs(nose.x - ((left_shoulder.x + right_shoulder.x) / 2)) > 0.08: head_tilt_warnings += 1
            face_mesh_results = face_mesh.process(frame_rgb)
            if face_mesh_results.multi_face_landmarks:
                face_landmarks = face_mesh_results.multi_face_landmarks[0].landmark
                l_iris_center = np.mean([(face_landmarks[i].x, face_landmarks[i].y) for i in LEFT_IRIS], axis=0)
                l_corner_left = np.array([face_landmarks[LEFT_EYE_CORNERS[0]].x, face_landmarks[LEFT_EYE_CORNERS[0]].y]); l_corner_right = np.array([face_landmarks[LEFT_EYE_CORNERS[1]].x, face_landmarks[LEFT_EYE_CORNERS[1]].y])
                eye_width = np.linalg.norm(l_corner_right - l_corner_left); ratio = np.linalg.norm(l_iris_center - l_corner_left) / eye_width if eye_width > 0 else 0.5
                if 2.70 < ratio < 3.10: forward_gaze_frames += 1
    finally: cap.release()
    video_analysis_result = {"dominant_emotion": "N/A", "emotion_distribution": {}}; gaze_stability = 0; avg_posture_score = 0; tilt_status = "Good"
    if emotions: emotion_counts = Counter(emotions); video_analysis_result = {"dominant_emotion": emotion_counts.most_common(1)[0][0], "emotion_distribution": {k: round(v / len(emotions) * 100) for k, v in emotion_counts.items()}}
    if total_frames > 0: gaze_stability = round((forward_gaze_frames / total_frames) * 100); tilt_status = "Needs Improvement" if (head_tilt_warnings / total_frames) * 100 > 30 else "Good"
    if posture_scores: avg_posture_score = round(np.mean(posture_scores))
    return {"video_analysis": video_analysis_result, "eye_contact": {"gaze_stability": gaze_stability}, "posture": {"score": avg_posture_score, "tilt_status": tilt_status}}

# --- API Endpoints ---
@app.post("/analyze")
async def analyze_performance(media_file: UploadFile = File(...), user=Depends(get_user)):
    temp_video_path = f"temp_{media_file.filename}"; temp_audio_path = "temp_extracted_audio.wav"
    try:
        with open(temp_video_path, "wb") as f: f.write(await media_file.read())
        subprocess.run(['ffmpeg', '-i', temp_video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', temp_audio_path, '-y'], check=True, capture_output=True)
        segments, _ = whisper_model.transcribe(temp_audio_path, beam_size=5, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
        transcript = "".join(segment.text for segment in segments).strip()
        duration = librosa.get_duration(path=temp_audio_path)
        
        video_features = analyze_video_features(temp_video_path)
        
        analysis_data = {
            "transcription": transcript or "No speech detected.", "duration": round(duration, 2),
            "wpm": calculate_wpm(transcript, duration), "filler_words": count_filler_words(transcript),
            "sentiment": analyze_sentiment(transcript), "audio_quality": analyze_audio_quality(temp_audio_path),
            **video_features
        }
        
        # Save to Supabase
        db_payload = { "user_id": user.id, **analysis_data }
        supabase.table("speech_sessions").insert(db_payload).execute()
        return analysis_data
    except Exception as e: raise HTTPException(500, f"An unexpected error occurred: {e}")
    finally:
        if os.path.exists(temp_video_path): os.remove(temp_video_path)
        if os.path.exists(temp_audio_path): os.remove(temp_audio_path)

@app.get("/sessions")
def get_sessions(user=Depends(get_user)):
    try:
        response = supabase.table("speech_sessions").select("*").eq("user_id", user.id).order("created_at", desc=True).execute()
        return response.data
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

