import React, { useState, useRef } from 'react';

// A new component to display the detailed analysis report
const AnalysisReport = ({ data }) => {
  if (!data) return null;

  const fillerWordCount = Object.values(data.filler_words).reduce((a, b) => a + b, 0);

  const getWpmStyle = (wpm) => {
    if (wpm < 130) return { color: '#f0ad4e' }; // Slow - Orange
    if (wpm > 160) return { color: '#d9534f' }; // Fast - Red
    return { color: '#5cb85c' }; // Good - Green
  };
  
  return (
    <div style={{ 
      marginTop: '20px', 
      padding: '20px', 
      border: '1px solid #ccc', 
      borderRadius: '8px', 
      textAlign: 'left',
      backgroundColor: '#f9f9f9'
    }}>
      <h3 style={{ textAlign: 'center', marginBottom: '20px' }}>Analysis Report</h3>
      
      {/* Key Metrics */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '15px', marginBottom: '20px' }}>
        <div style={{ padding: '10px', border: '1px solid #ddd', borderRadius: '5px' }}>
          <strong>Speaking Pace (WPM)</strong>
          <p style={{ ...getWpmStyle(data.wpm), fontSize: '24px', margin: '5px 0 0' }}>{data.wpm}</p>
        </div>
        <div style={{ padding: '10px', border: '1px solid #ddd', borderRadius: '5px' }}>
          <strong>Total Filler Words</strong>
          <p style={{ fontSize: '24px', margin: '5px 0 0', color: fillerWordCount > 5 ? '#d9534f' : '#5cb85c' }}>{fillerWordCount}</p>
        </div>
        <div style={{ padding: '10px', border: '1px solid #ddd', borderRadius: '5px' }}>
          <strong>Confidence Score</strong>
          <p style={{ fontSize: '24px', margin: '5px 0 0' }}>{data.audio_quality.confidence_score}%</p>
        </div>
      </div>
      
      {/* Transcription */}
      <div>
        <strong>Transcription:</strong>
        <p style={{ whiteSpace: 'pre-wrap', backgroundColor: '#fff', padding: '10px', borderRadius: '5px', border: '1px solid #ddd' }}>{data.transcription || "No speech detected."}</p>
      </div>

      {/* Filler Words Breakdown */}
      {fillerWordCount > 0 && (
         <div style={{marginTop: '15px'}}>
            <strong>Filler Words Used:</strong>
            <ul>
              {Object.entries(data.filler_words).map(([word, count]) => (
                <li key={word}>{word}: {count}</li>
              ))}
            </ul>
        </div>
      )}

       {/* Audio Quality */}
       <div style={{marginTop: '15px'}}>
          <strong>Audio Quality:</strong>
          <ul>
            <li>Pitch Variation: {data.audio_quality.pitch_variation}</li>
            <li>Volume Consistency: {data.audio_quality.volume_consistency}%</li>
          </ul>
      </div>

    </div>
  );
};


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
      mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      
      audioChunksRef.current = []; // Clear previous chunks

      mediaRecorderRef.current.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data);
      };

      mediaRecorderRef.current.onstop = () => {
        // This is the raw blob, which is perfectly playable and sendable
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        setAudioBlob(blob);
        audioChunksRef.current = [];
        // Stop the mic track to turn off the browser's recording indicator
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setAudioBlob(null);
      setAnalysisResult(null);
    } catch (error) {
      console.error("Error accessing microphone:", error);
      alert("Could not access microphone. Please ensure permissions are granted.");
    }
  };

  const handleStopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleAnalyze = async () => {
    if (!audioBlob) {
      alert("Please record some audio first.");
      return;
    }
    
    setIsLoading(true);
    setAnalysisResult(null);

    const formData = new FormData();
    // We send a .webm file now, which is fine.
    formData.append('audio_file', audioBlob, 'recording.webm');

    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Analysis failed on the server.');
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
    <div style={{ maxWidth: '700px', margin: 'auto', padding: '40px', fontFamily: 'sans-serif', textAlign: 'center' }}>
      <h1>AI Speech Coach 🎙️</h1>
      <p>Record your voice, then get instant feedback on your delivery.</p>
      
      <div style={{ margin: '20px 0' }}>
        <button 
          onClick={isRecording ? handleStopRecording : handleStartRecording}
          style={{ 
            padding: '15px 30px', 
            fontSize: '18px', 
            cursor: 'pointer',
            backgroundColor: isRecording ? '#d9534f' : '#5cb85c',
            color: 'white',
            border: 'none',
            borderRadius: '50px',
            minWidth: '200px',
            transition: 'background-color 0.3s'
          }}
        >
          {isRecording ? 'Stop Recording' : 'Start Recording'}
        </button>
      </div>
      
      {audioBlob && !isRecording && (
        <div style={{ margin: '20px 0', padding: '20px', backgroundColor: '#f0f0f0', borderRadius: '10px' }}>
          <h4>Your Recording:</h4>
          <audio controls src={URL.createObjectURL(audioBlob)} style={{width: '100%'}} />
          <br />
          <button 
            onClick={handleAnalyze} 
            disabled={isLoading}
            style={{ 
              padding: '10px 20px', 
              fontSize: '16px', 
              marginTop: '15px',
              cursor: 'pointer',
              backgroundColor: '#0275d8',
              color: 'white',
              border: 'none',
              borderRadius: '5px'
            }}
          >
            {isLoading ? 'Analyzing...' : 'Analyze My Speech'}
          </button>
        </div>
      )}
      
      {isLoading && <p>Hold on, analyzing your speech...</p>}
      
      <AnalysisReport data={analysisResult} />
    </div>
  );
}

export default App;
