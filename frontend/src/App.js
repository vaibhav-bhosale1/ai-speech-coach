import React, { useState, useRef } from 'react';

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const handleStartRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      
      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        setAudioBlob(blob);
        audioChunksRef.current = [];
        // Get the microphone track and stop it to turn off the mic indicator
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setAudioBlob(null); // Clear previous recording
      setAnalysisResult(null); // Clear previous results
    } catch (error) {
      console.error("Error accessing microphone:", error);
      alert("Could not access microphone. Please ensure permissions are granted.");
    }
  };

  const handleStopRecording = () => {
    mediaRecorderRef.current.stop();
    setIsRecording(false);
  };

  const handleAnalyze = async () => {
    if (!audioBlob) {
      alert("Please record some audio first.");
      return;
    }
    
    setIsLoading(true);
    setAnalysisResult(null);

    const formData = new FormData();
    formData.append('audio_file', audioBlob, 'recording.wav');

    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const result = await response.json();
      setAnalysisResult(result);
    } catch (error) {
      console.error("Error analyzing audio:", error);
      alert("An error occurred while analyzing the audio.");
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div style={{ padding: '40px', fontFamily: 'sans-serif', textAlign: 'center' }}>
      <h1>AI Speech Coach 🎙️</h1>
      <p>Record your voice and get instant feedback.</p>
      
      <div style={{ margin: '20px 0' }}>
        <button 
          onClick={isRecording ? handleStopRecording : handleStartRecording}
          style={{ 
            padding: '10px 20px', 
            fontSize: '16px', 
            cursor: 'pointer',
            backgroundColor: isRecording ? '#f44336' : '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: '5px'
          }}
        >
          {isRecording ? 'Stop Recording' : 'Start Recording'}
        </button>
      </div>
      
      {audioBlob && !isRecording && (
        <div style={{ margin: '20px 0' }}>
          <audio controls src={URL.createObjectURL(audioBlob)} />
          <br />
          <button 
            onClick={handleAnalyze} 
            disabled={isLoading}
            style={{ padding: '10px 20px', fontSize: '16px', marginTop: '10px' }}
          >
            {isLoading ? 'Analyzing...' : 'Analyze My Speech'}
          </button>
        </div>
      )}
      
      {isLoading && <p>Processing your speech, please wait...</p>}
      
      {analysisResult && (
        <div style={{ marginTop: '20px', padding: '20px', border: '1px solid #ccc', borderRadius: '8px', textAlign: 'left' }}>
          <h3>Analysis Confirmation:</h3>
          <pre>{JSON.stringify(analysisResult, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default App;