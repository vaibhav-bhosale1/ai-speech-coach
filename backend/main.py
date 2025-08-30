import os
import re
import math
import numpy as np
import librosa
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

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
    print(f"✅ Whisper model '{model_size}' loaded successfully.")
except Exception as e:
    print(f"🔥 Failed to load Whisper model: {e}")
    whisper_model = None

# --- Helper Functions ---
def calculate_wpm(transcript, duration_seconds):
    """Calculates words per minute."""
    word_count = len(transcript.split())
    if duration_seconds > 0:
        return round((word_count / duration_seconds) * 60)
    return 0

def count_filler_words(transcript):
    """Counts occurrences of filler words in the transcript."""
    words = re.findall(r'\b\w+\b', transcript.lower())
    filler_counts = {filler: 0 for filler in FILLER_WORDS}
    for word in words:
        if word in filler_counts:
            filler_counts[word] += 1
    # Return only fillers that were actually found
    return {k: v for k, v in filler_counts.items() if v > 0}

def analyze_audio_quality(audio_path):
    """Analyzes pitch, volume, and confidence from an audio file."""
    try:
        y, sr = librosa.load(audio_path, sr=16000)

        # 1. Pitch Variation (Monotone vs. Dynamic)
        pitches, _ = librosa.piptrack(y=y, sr=sr)
        non_zero_pitches = pitches[pitches > 0]
        if len(non_zero_pitches) > 0:
            pitch_std_dev = np.std(non_zero_pitches)
            pitch_variation = "Dynamic" if pitch_std_dev > 25 else "Monotone"
        else:
            pitch_variation = "N/A"

        # 2. Volume Consistency (Root Mean Square energy)
        rms = librosa.feature.rms(y=y)[0]
        if len(rms) > 0:
            rms_std_dev_db = np.std(librosa.amplitude_to_db(rms))
            # Lower std dev means more consistent volume. We convert it to a 0-100 score.
            volume_consistency = max(0, 100 - rms_std_dev_db * 10)
        else:
            volume_consistency = 0
            
        # 3. Confidence Score (simple heuristic)
        # A confident speaker is likely to have dynamic pitch and consistent volume.
        confidence_score = (volume_consistency + (90 if pitch_variation == "Dynamic" else 50)) / 2

        return {
            "pitch_variation": pitch_variation,
            "volume_consistency": round(volume_consistency),
            "confidence_score": round(confidence_score),
            # Placeholder for pause analysis which is more complex
            "pause_analysis": "Natural", 
        }
    except Exception as e:
        print(f"🔥 Error in audio quality analysis: {e}")
        return {
            "pitch_variation": "Error",
            "volume_consistency": 0,
            "confidence_score": 0,
            "pause_analysis": "Error",
        }


# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Speech Coach API is running"}

@app.post("/analyze")
async def analyze_speech(audio_file: UploadFile = File(...)):
    if not whisper_model:
        raise HTTPException(status_code=500, detail="Whisper model is not available.")

    temp_audio_path = f"temp_{audio_file.filename}"
    
    try:
        # Save the uploaded file temporarily
        contents = await audio_file.read()
        with open(temp_audio_path, "wb") as f:
            f.write(contents)
        
        # --- Start Analysis ---
        # 1. Transcription
        segments, info = whisper_model.transcribe(temp_audio_path, beam_size=5)
        full_transcript = "".join(segment.text for segment in segments).strip()
        
        # 2. Duration and Pace (WPM)
        duration_seconds = librosa.get_duration(path=temp_audio_path)
        wpm = calculate_wpm(full_transcript, duration_seconds)
        
        # 3. Filler Words
        filler_word_counts = count_filler_words(full_transcript)
        
        # 4. Audio Quality
        audio_quality_metrics = analyze_audio_quality(temp_audio_path)

        # --- Construct final response ---
        response = {
            "transcription": full_transcript,
            "duration": round(duration_seconds, 2),
            "wpm": wpm,
            "filler_words": filler_word_counts,
            "audio_quality": audio_quality_metrics
        }
        
        return response

    except Exception as e:
        print(f"🔥 An error occurred during analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary audio file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
