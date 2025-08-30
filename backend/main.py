import os
import re
import math
import numpy as np
import librosa
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Constants ---
FILLER_WORDS = [
    'um', 'uh', 'er', 'ah', 'like', 'okay', 'right', 'so', 'you know', 
    'basically', 'actually', 'literally', 'well', 'i mean'
]

# --- App Setup ---
app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AI Model Loading ---
try:
    model_size = "base.en"
    whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
    sentiment_analyzer = SentimentIntensityAnalyzer()
    print(f"✅ Whisper model '{model_size}' and VADER sentiment analyzer loaded successfully.")
except Exception as e:
    print(f"🔥 Failed to load an AI model: {e}")
    whisper_model = None
    sentiment_analyzer = None

# --- Helper Functions ---
def calculate_wpm(transcript, duration_seconds):
    word_count = len(transcript.split())
    if duration_seconds > 0:
        return round((word_count / duration_seconds) * 60)
    return 0

def count_filler_words(transcript):
    words = re.findall(r'\b\w+\b', transcript.lower())
    filler_counts = {filler: 0 for filler in FILLER_WORDS}
    for word in words:
        if word in filler_counts:
            filler_counts[word] += 1
    return {k: v for k, v in filler_counts.items() if v > 0}

def analyze_audio_quality(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        non_zero_pitches = pitches[pitches > 0]
        pitch_std_dev = np.std(non_zero_pitches) if len(non_zero_pitches) > 0 else 0.0
        pitch_variation = "Dynamic" if pitch_std_dev > 25 else "Monotone"
        
        rms = librosa.feature.rms(y=y)[0]
        rms_std_dev_db = np.std(librosa.amplitude_to_db(rms)) if len(rms) > 0 else 50.0
        volume_consistency = max(0, 100 - rms_std_dev_db * 10)
        
        confidence_score = (volume_consistency + (90 if pitch_variation == "Dynamic" else 50)) / 2

        return {
            "pitch_variation": pitch_variation,
            "volume_consistency": round(volume_consistency),
            "confidence_score": round(confidence_score),
            "pause_analysis": "Natural",
        }
    except Exception as e:
        print(f"🔥 Error in audio quality analysis: {e}")
        return {"pitch_variation": "Error", "volume_consistency": 0, "confidence_score": 0, "pause_analysis": "Error"}

def analyze_sentiment(transcript):
    """Analyzes the sentiment of the transcript using VADER."""
    if not sentiment_analyzer:
        return {"pos": 0, "neu": 0, "neg": 0, "compound": 0, "label": "N/A"}
    
    scores = sentiment_analyzer.polarity_scores(transcript)
    compound = scores['compound']
    label = "Neutral"
    if compound > 0.05:
        label = "Positive"
    elif compound < -0.05:
        label = "Negative"
        
    return {
        "pos": round(scores['pos'] * 100),
        "neu": round(scores['neu'] * 100),
        "neg": round(scores['neg'] * 100),
        "compound": scores['compound'],
        "label": label
    }

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Speech Coach API is running"}

@app.post("/analyze")
async def analyze_speech(audio_file: UploadFile = File(...)):
    if not whisper_model or not sentiment_analyzer:
        raise HTTPException(status_code=500, detail="An AI model is not available.")

    temp_audio_path = f"temp_{audio_file.filename}"
    
    try:
        with open(temp_audio_path, "wb") as f:
            f.write(await audio_file.read())
        
        segments, info = whisper_model.transcribe(temp_audio_path, beam_size=5)
        full_transcript = "".join(segment.text for segment in segments).strip()
        
        duration_seconds = librosa.get_duration(path=temp_audio_path)
        wpm = calculate_wpm(full_transcript, duration_seconds)
        filler_word_counts = count_filler_words(full_transcript)
        audio_quality_metrics = analyze_audio_quality(temp_audio_path)
        sentiment_scores = analyze_sentiment(full_transcript)

        response = {
            "transcription": full_transcript,
            "duration": round(duration_seconds, 2),
            "wpm": wpm,
            "filler_words": filler_word_counts,
            "audio_quality": audio_quality_metrics,
            "sentiment": sentiment_scores,
        }
        return response

    except Exception as e:
        print(f"🔥 An error occurred during analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

