import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

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

@app.get("/")
def read_root():
    return {"message": "Speech Coach API is running"}

@app.post("/analyze")
async def analyze_speech(audio_file: UploadFile = File(...)):
    """
    Receives an audio file, saves it temporarily, and returns its size.
    """
    # Define a temporary path to save the file
    temp_audio_path = "temp_audio.wav"
    
    # Read the contents of the uploaded file
    contents = await audio_file.read()
    
    # Save the file to the temporary path
    with open(temp_audio_path, "wb") as f:
        f.write(contents)
        
    file_size = os.path.getsize(temp_audio_path)
    
    # For now, we just confirm receipt. The actual analysis will be in Phase 3.
    # The temporary file will be processed and deleted in later phases.
    
    return {
        "message": f"Audio file '{audio_file.filename}' received successfully.",
        "file_size_bytes": file_size
    }