import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel

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
# Load the model once when the application starts.
# "base.en" is a good starting point. For better accuracy, consider "small.en".
# Using device="cpu" and compute_type="int8" for better performance on non-GPU servers.
try:
    model_size = "base.en"
    whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print(f"✅ Whisper model '{model_size}' loaded successfully.")
except Exception as e:
    print(f"🔥 Failed to load Whisper model: {e}")
    whisper_model = None

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "Speech Coach API is running"}

@app.post("/analyze")
async def analyze_speech(audio_file: UploadFile = File(...)):
    if not whisper_model:
        raise HTTPException(status_code=500, detail="Whisper model is not available.")

    temp_audio_path = "temp_audio.wav"
    
    try:
        # Save the uploaded file temporarily
        contents = await audio_file.read()
        with open(temp_audio_path, "wb") as f:
            f.write(contents)
        
        # Transcribe the audio file
        segments, info = whisper_model.transcribe(temp_audio_path, beam_size=5)
        
        print(f"Detected language '{info.language}' with probability {info.language_probability}")
        
        # Concatenate segments to form the full transcript
        full_transcript = "".join(segment.text for segment in segments).strip()

        return {"transcription": full_transcript}

    except Exception as e:
        print(f"🔥 An error occurred during analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up: remove the temporary audio file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)