import React, { useState, useRef } from 'react';

// --- SVG Icons ---
const MicIcon = ({ className }) => (
  <svg className={className} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"></path>
    <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"></path>
  </svg>
);

const LoaderIcon = () => (
    <svg className="animate-spin h-5 w-5 mr-3" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 2.99988V5.99988" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M12 18.0001V21.0001" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M5.63574 5.63623L7.75674 7.75723" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M16.2422 16.2427L18.3632 18.3637" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M3 12.0001H6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M18 12.0001H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M5.63574 18.3637L7.75674 16.2427" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      <path d="M16.2422 7.75723L18.3632 5.63623" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
);

// --- UI Components ---
const StatCard = ({ title, value, subtext, colorClass = 'text-white' }) => (
  <div className="bg-gray-700 p-4 rounded-lg text-center shadow-md">
    <h3 className="text-sm text-gray-400 font-medium uppercase tracking-wider">{title}</h3>
    <p className={`text-3xl font-bold ${colorClass}`}>{value}</p>
    {subtext && <p className="text-xs text-gray-500">{subtext}</p>}
  </div>
);

const AnalysisReport = ({ data }) => {
  if (!data) return null;

  const fillerWordCount = data.filler_words ? Object.values(data.filler_words).reduce((a, b) => a + b, 0) : 0;
  
  const getWpmColor = (wpm) => {
    if (wpm < 130) return 'text-yellow-400';
    if (wpm > 160) return 'text-red-400';
    return 'text-green-400';
  };
  
  const sentimentLabel = data.sentiment?.label || 'Neutral';
  const getSentimentColor = (label) => {
      if(label === 'Positive') return 'text-green-400';
      if(label === 'Negative') return 'text-red-400';
      return 'text-yellow-400';
  }

  return (
    <div className="w-full max-w-4xl mx-auto bg-gray-800 rounded-xl shadow-lg p-6 space-y-6">
      <h2 className="text-2xl font-bold text-center">Analysis Report</h2>
      
      {/* Key Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard title="Speaking Pace" value={data.wpm || 0} subtext="WPM" colorClass={getWpmColor(data.wpm)} />
        <StatCard title="Filler Words" value={fillerWordCount} colorClass={fillerWordCount > 5 ? 'text-red-400' : 'text-green-400'} />
        <StatCard title="Confidence" value={`${data.audio_quality?.confidence_score || 0}%`} />
        <StatCard title="Sentiment" value={sentimentLabel} colorClass={getSentimentColor(sentimentLabel)} />
      </div>
      
      {/* Transcription */}
      <div className="bg-gray-700 p-4 rounded-lg">
        <h3 className="text-sm text-gray-400 font-medium mb-2">Transcription</h3>
        <p className="text-gray-300 whitespace-pre-wrap font-mono text-sm leading-relaxed">{data.transcription || "No speech was detected."}</p>
      </div>

      {/* Details Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {fillerWordCount > 0 && (
          <div className="bg-gray-700 p-4 rounded-lg">
            <h3 className="text-sm text-gray-400 font-medium mb-2">Filler Words Breakdown</h3>
            <ul className="text-gray-300 text-sm space-y-1">
              {Object.entries(data.filler_words).map(([word, count]) => (
                <li key={word}><span className="font-semibold capitalize">{word}:</span> {count}</li>
              ))}
            </ul>
          </div>
        )}
         <div className="bg-gray-700 p-4 rounded-lg">
            <h3 className="text-sm text-gray-400 font-medium mb-2">Audio Quality</h3>
            <ul className="text-gray-300 text-sm space-y-1">
              <li>Pitch: <span className="font-semibold">{data.audio_quality?.pitch_variation || 'N/A'}</span></li>
              <li>Volume: <span className="font-semibold">{data.audio_quality?.volume_consistency || 0}% Consistent</span></li>
            </ul>
        </div>
        <div className="bg-gray-700 p-4 rounded-lg">
            <h3 className="text-sm text-gray-400 font-medium mb-2">Sentiment Analysis</h3>
            <ul className="text-gray-300 text-sm space-y-1">
              <li>Positive: <span className="font-semibold text-green-400">{data.sentiment?.pos || 0}%</span></li>
              <li>Neutral: <span className="font-semibold text-yellow-400">{data.sentiment?.neu || 0}%</span></li>
              <li>Negative: <span className="font-semibold text-red-400">{data.sentiment?.neg || 0}%</span></li>
            </ul>
        </div>
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
      
      audioChunksRef.current = [];
      mediaRecorderRef.current.ondataavailable = event => audioChunksRef.current.push(event.data);

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        setAudioBlob(blob);
        audioChunksRef.current = [];
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setAudioBlob(null);
      setAnalysisResult(null);
    } catch (error) {
      console.error("Mic error:", error);
      alert("Microphone access denied. Please allow microphone access in your browser settings.");
    }
  };

  const handleStopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleAnalyze = async () => {
    if (!audioBlob) return;
    setIsLoading(true);
    setAnalysisResult(null);

    const formData = new FormData();
    formData.append('audio_file', audioBlob, 'recording.webm');

    try {
      const response = await fetch('http://localhost:8000/analyze', { method: 'POST', body: formData });
      if (!response.ok) throw new Error('Server analysis failed.');
      const result = await response.json();
      setAnalysisResult(result);
    } catch (error) {
      console.error("Analysis error:", error);
      alert("An error occurred during analysis.");
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div className="bg-gray-900 text-white min-h-screen flex flex-col items-center justify-center p-4 font-sans">
      <div className="w-full max-w-4xl space-y-8">
        <header className="text-center">
          <h1 className="text-4xl md:text-5xl font-bold">AI Speech Coach</h1>
          <p className="text-gray-400 mt-2">Get instant, AI-powered feedback on your public speaking.</p>
        </header>
        
        <main className="flex flex-col items-center space-y-6">
          <button 
            onClick={isRecording ? handleStopRecording : handleStartRecording}
            className={`flex items-center justify-center w-40 h-40 rounded-full text-white font-semibold transition-all duration-300 ease-in-out focus:outline-none focus:ring-4
              ${isRecording 
                ? 'bg-red-600 hover:bg-red-700 focus:ring-red-500 animate-pulse' 
                : 'bg-blue-600 hover:bg-blue-700 focus:ring-blue-500'}`
              }
          >
            <MicIcon className="w-16 h-16" />
          </button>
          <p className="text-lg">{isRecording ? 'Recording in progress...' : 'Press the button to start recording'}</p>
          
          {audioBlob && !isRecording && (
            <div className="w-full max-w-md bg-gray-800 p-4 rounded-lg space-y-4">
              <h3 className="font-semibold text-center">Your Recording</h3>
              <audio controls src={URL.createObjectURL(audioBlob)} className="w-full" />
              <button 
                onClick={handleAnalyze} 
                disabled={isLoading}
                className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-500 disabled:cursor-not-allowed text-white font-bold py-3 px-4 rounded-lg transition-colors flex items-center justify-center"
              >
                {isLoading ? <><LoaderIcon /> Analyzing...</> : 'Analyze My Speech'}
              </button>
            </div>
          )}

          {isLoading && !analysisResult && <p className="text-center">Hold on, the AI is analyzing your speech...</p>}
        </main>

        {analysisResult && <AnalysisReport data={analysisResult} />}
        
        <footer className="text-center text-gray-500 text-sm pt-8">
          <p>Powered by FastAPI, React & Faster Whisper</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
