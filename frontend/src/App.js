import React, { useState, useRef, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, Radar, PieChart, Pie, Cell, Legend } from 'recharts';

// --- SVG Icons ---
const MicIcon = ({ className }) => (<svg className={className} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"></path><path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"></path></svg>);
const LoaderIcon = () => (<svg className="animate-spin h-5 w-5 mr-3" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2.99988V5.99988" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/><path d="M12 18.0001V21.0001" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/><path d="M5.63574 5.63623L7.75674 7.75723" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/><path d="M16.2422 16.2427L18.3632 18.3637" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/><path d="M3 12.0001H6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/><path d="M18 12.0001H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/><path d="M5.63574 18.3637L7.75674 16.2427" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/><path d="M16.2422 7.75723L18.3632 5.63623" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>);
const ClipboardIcon = ({ className }) => (<svg className={className} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path d="M7 3a1 1 0 000 2h6a1 1 0 100-2H7zM4 7a1 1 0 011-1h10a1 1 0 110 2H5a1 1 0 01-1-1zM2 11a2 2 0 012-2h12a2 2 0 012 2v3a2 2 0 01-2 2H4a2 2 0 01-2-2v-3z"></path></svg>);
const CheckIcon = ({ className }) => (<svg className={className} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd"></path></svg>);
const DownloadIcon = ({ className }) => (<svg className={className} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clipRule="evenodd"></path></svg>);

// --- Reusable UI Components ---
const StatCard = ({ title, value, subtext, colorClass = 'text-gray-900' }) => (
    <div className="bg-white border border-gray-200 p-6 rounded-lg text-center shadow-sm hover:shadow-md transition-all duration-200">
        <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">{title}</h3>
        <p className={`text-3xl font-bold ${colorClass} mb-1`}>{value}</p>
        {subtext && !isNaN(parseFloat(value)) && <p className="text-xs text-gray-400">{subtext}</p>}
    </div>
);

const ChartPlaceholder = ({ message }) => (
    <div className="bg-gray-50 border border-gray-200 p-8 rounded-lg flex items-center justify-center h-full min-h-[280px]">
        <p className="text-gray-500 text-center">{message}</p>
    </div>
);

const ReportSection = ({ title, children }) => (
    <div className="bg-white border border-gray-200 p-6 rounded-lg shadow-sm">
        <h3 className="text-lg font-semibold text-gray-900 mb-6 pb-3 border-b border-gray-100">
            {title}
        </h3>
        {children}
    </div>
);

// --- Chart Components ---
const EmotionChart = ({ data }) => {
    const COLORS = { 
        Happy: '#059669', 
        Sad: '#2563EB', 
        Angry: '#DC2626', 
        Neutral: '#6B7280', 
        Surprise: '#D97706', 
        Fear: '#7C3AED', 
        Disgust: '#DB2777', 
        Contempt: '#78716c' 
    };
    const chartData = data ? Object.entries(data).map(([name, value]) => ({ name, value })) : [];
    if (chartData.length === 0) return <ChartPlaceholder message="No facial emotion data detected." />;
    return (
        <ResponsiveContainer width="100%" height={280}>
            <PieChart>
                <Pie 
                    data={chartData} 
                    dataKey="value" 
                    nameKey="name" 
                    cx="50%" 
                    cy="50%" 
                    outerRadius={100} 
                    labelLine={false}
                    strokeWidth={2}
                    stroke="#ffffff"
                >
                    {chartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[entry.name] || '#6B7280'} />
                    ))}
                </Pie>
                <Tooltip 
                    contentStyle={{
                        backgroundColor: '#ffffff', 
                        border: '1px solid #e5e7eb', 
                        borderRadius: '8px',
                        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                    }}
                />
                <Legend />
            </PieChart>
        </ResponsiveContainer>
    );
};

const FillerWordsChart = ({ data }) => {
    const chartData = data ? Object.entries(data).map(([name, count]) => ({ name, count })) : [];
    if (chartData.length === 0) return null; 
    return (
        <ResponsiveContainer width="100%" height={280}>
            <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 20, left: 40, bottom: 5 }}>
                <XAxis type="number" hide />
                <YAxis 
                    type="category" 
                    dataKey="name" 
                    width={60} 
                    stroke="#6b7280" 
                    tick={{ fill: '#374151', fontSize: 12 }} 
                    interval={0}
                />
                <Tooltip 
                    cursor={{fill: 'rgba(59, 130, 246, 0.1)'}} 
                    contentStyle={{
                        backgroundColor: '#ffffff', 
                        border: '1px solid #e5e7eb', 
                        borderRadius: '8px',
                        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                    }}
                />
                <Bar dataKey="count" fill="#3B82F6" barSize={24} radius={[0, 4, 4, 0]} />
            </BarChart>
        </ResponsiveContainer>
    );
};

const AudioQualityChart = ({ data }) => {
    if(data?.pitch_variation === 'N/A') return <ChartPlaceholder message="Audio was silent or too quiet for quality analysis."/>
    const chartData = [
        { subject: 'Confidence', A: data?.confidence_score || 0, fullMark: 100 },
        { subject: 'Volume', A: data?.volume_consistency || 0, fullMark: 100 },
        { subject: 'Pitch', A: data?.pitch_variation === 'Dynamic' ? 85 : 40, fullMark: 100 },
    ];
    return (
        <ResponsiveContainer width="100%" height={280}>
            <RadarChart cx="50%" cy="50%" outerRadius="85%" data={chartData}>
                <PolarGrid stroke="#d1d5db" />
                <PolarAngleAxis 
                    dataKey="subject" 
                    stroke="#6b7280" 
                    tick={{ fill: '#374151', fontSize: 12 }} 
                />
                <Radar 
                    name="Score" 
                    dataKey="A" 
                    stroke="#3B82F6" 
                    fill="#3B82F6" 
                    fillOpacity={0.2}
                    strokeWidth={2}
                    dot={{ fill: '#3B82F6', strokeWidth: 2, r: 4 }}
                />
            </RadarChart>
        </ResponsiveContainer>
    );
};

// --- Main Report Component ---
const AnalysisReport = ({ data, onGeneratePdf }) => {
  const [copied, setCopied] = useState(false);
  if (!data) return null;

  const handleCopy = () => {
    navigator.clipboard.writeText(data.transcription);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const noSpeechDetected = data.transcription === "No speech detected.";
  const fillerWordCount = !noSpeechDetected && data.filler_words ? Object.values(data.filler_words).reduce((a, b) => a + b, 0) : 0;
  
  const getWpmColor = (wpm) => (wpm > 0 && wpm < 130) ? 'text-amber-600' : wpm > 160 ? 'text-red-600' : 'text-green-600';
  const getPostureColor = (score) => (score < 80) ? 'text-red-600' : score < 95 ? 'text-amber-600' : 'text-green-600';
 
  return (
    <div id="analysisReport" className="w-full max-w-7xl mx-auto bg-white border border-gray-200 rounded-lg shadow-sm p-8 space-y-8">
      <div className="flex justify-between items-center border-b border-gray-200 pb-6">
        <div>
          <h2 className="text-3xl font-bold text-gray-900">Analysis Report</h2>
          <p className="text-gray-600 mt-2">Comprehensive analysis of your performance</p>
        </div>
        <button 
          onClick={onGeneratePdf} 
          className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 px-6 rounded-lg transition-colors duration-200 flex items-center space-x-2 shadow-sm"
        >
          <DownloadIcon className="w-5 h-5" />
          <span>Download PDF</span>
        </button>
      </div>
      
      <ReportSection title="Performance Metrics">
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            <StatCard title="Speaking Pace" value={noSpeechDetected ? 'N/A' : (data.wpm || 0)} subtext="WPM" colorClass={getWpmColor(data.wpm)} />
            <StatCard title="Filler Words" value={noSpeechDetected ? 'N/A' : fillerWordCount} colorClass={fillerWordCount > 5 ? 'text-red-600' : 'text-green-600'} />
            <StatCard title="Sentiment" value={noSpeechDetected ? 'N/A' : (data.sentiment?.label || 'N/A')} />
            <StatCard title="Dom. Emotion" value={data.video_analysis?.dominant_emotion ?? 'N/A'} />
            <StatCard title="Eye Contact" value={data.eye_contact?.gaze_stability ?? 'N/A'} subtext="% stability" />
            <StatCard title="Posture Score" value={data.posture?.score ?? 'N/A'} subtext="/ 100" colorClass={getPostureColor(data.posture?.score)} />
        </div>
      </ReportSection>

      <ReportSection title="Speech Transcription">
        <div className="relative">
            <button 
              onClick={handleCopy} 
              className="absolute -top-2 right-0 bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm px-4 py-2 rounded-md transition-colors duration-200 flex items-center space-x-2 border border-gray-300"
            >
                {copied ? (
                  <>
                    <CheckIcon className="w-4 h-4 text-green-600" />
                    <span className="text-green-600">Copied!</span>
                  </>
                ) : (
                  <>
                    <ClipboardIcon className="w-4 h-4" />
                    <span>Copy Text</span>
                  </>
                )}
            </button>
            <div className="bg-gray-50 border border-gray-200 p-6 rounded-lg mt-4">
              <p className="text-gray-700 whitespace-pre-wrap font-mono text-sm leading-relaxed max-h-48 overflow-y-auto">
                {data.transcription || ""}
              </p>
            </div>
        </div>
      </ReportSection>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <ReportSection title="Facial Emotion Distribution">
          <EmotionChart data={data.video_analysis?.emotion_distribution} />
        </ReportSection>
        <ReportSection title="Audio Quality Analysis">
            {noSpeechDetected ? (
              <ChartPlaceholder message="Audio metrics are unavailable as no speech was detected." />
            ) : (
              <AudioQualityChart data={data.audio_quality} />
            )}
        </ReportSection>
        {!noSpeechDetected && fillerWordCount > 0 && 
            <ReportSection title="Filler Words Breakdown">
                <FillerWordsChart data={data.filler_words} />
            </ReportSection>
        }
      </div>
    </div>
  );
};

// --- Main App Component ---
function App() {
    const [isRecording, setIsRecording] = useState(false);
    const [stream, setStream] = useState(null);
    const [mediaBlob, setMediaBlob] = useState(null);
    const [analysisResult, setAnalysisResult] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [statusText, setStatusText] = useState('Press the record button to start your session');
    
    const mediaRecorderRef = useRef(null);
    const videoRef = useRef(null);
    const recordedVideoRef = useRef(null);

    useEffect(() => { 
        if (stream && videoRef.current) { 
            videoRef.current.srcObject = stream; 
        } 
    }, [stream]);

    const handleGeneratePdf = async () => {
        const reportElement = document.getElementById('analysisReport');
        if (!reportElement) return;
        setStatusText('Generating your PDF report...');
        
        try {
            // Simulate PDF generation (you'll need to add html2canvas and jsPDF imports)
            const originalBg = reportElement.style.backgroundColor;
            reportElement.style.backgroundColor = '#ffffff';
            // const canvas = await html2canvas(reportElement, { scale: 2, useCORS: true, backgroundColor: '#ffffff' });
            // reportElement.style.backgroundColor = originalBg;
            // const imgData = canvas.toDataURL('image/png');
            // const pdf = new jsPDF({ orientation: 'p', unit: 'px', format: [canvas.width, canvas.height] });
            // pdf.addImage(imgData, 'PNG', 0, 0, canvas.width, canvas.height);
            // pdf.save(`Speech-Analysis-Report-${new Date().toISOString().slice(0,10)}.pdf`);
            setStatusText('PDF would be downloaded (requires html2canvas and jsPDF libraries).');
        } catch (error) {
            setStatusText('PDF generation failed.');
        }
    };

    const handleStartRecording = async () => {
        try {
            const mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: true });
            setStream(mediaStream);
            setIsRecording(true);
            setMediaBlob(null);
            setAnalysisResult(null);
            setStatusText('Recording in progress... Speak clearly and maintain good posture');
            const chunks = [];
            mediaRecorderRef.current = new MediaRecorder(mediaStream, { mimeType: 'video/webm' });
            mediaRecorderRef.current.ondataavailable = event => chunks.push(event.data);
            mediaRecorderRef.current.onstop = () => {
                const blob = new Blob(chunks, { type: 'video/webm' });
                setMediaBlob(blob);
                mediaStream.getTracks().forEach(track => track.stop());
                setStream(null);
                setStatusText('Recording complete! Ready to analyze your performance');
            };
            mediaRecorderRef.current.start();
        } catch (error) {
            console.error("Media access error:", error);
            alert("Camera/Microphone access was denied. Please allow access in your browser settings and refresh the page.");
        }
    };

    const handleStopRecording = () => {
        if (mediaRecorderRef.current) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
        }
    };

    const handleAnalyze = async () => {
        if (!mediaBlob) return;
        setIsLoading(true);
        setAnalysisResult(null);
        setStatusText('AI is analyzing your performance... This may take a moment');
        const formData = new FormData();
        formData.append('media_file', mediaBlob, 'recording.webm');
        try {
            const response = await fetch('http://localhost:8000/analyze', { method: 'POST', body: formData });
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Server analysis failed.');
            }
            const result = await response.json();
            setAnalysisResult(result);
        } catch (error) {
            alert(`An error occurred during analysis: ${error.message}`);
        } finally {
            setIsLoading(false);
            setStatusText('Analysis complete! Review your detailed report below');
        }
    };

    return (
        <div className="min-h-screen bg-gray-50 text-gray-900">
            <div className="w-full max-w-7xl mx-auto px-6 py-8 space-y-12">
                <header className="text-center space-y-6">
                    <h1 className="text-5xl md:text-6xl font-bold text-gray-900">
                        AI Speech Coach
                    </h1>
                    <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
                        Transform your communication skills with instant, AI-powered analysis and personalized feedback
                    </p>
                </header>

                <main className="flex flex-col items-center space-y-8">
                    <div className="w-full max-w-4xl bg-white border border-gray-200 rounded-lg overflow-hidden aspect-video shadow-sm">
                        {isRecording ? (
                            <>
                                <video ref={videoRef} autoPlay muted className="w-full h-full object-cover"></video>
                                <div className="absolute top-4 left-4 bg-red-600 text-white px-3 py-1 rounded-full text-sm font-medium flex items-center">
                                    <div className="w-2 h-2 bg-white rounded-full mr-2 animate-pulse"></div>
                                    RECORDING
                                </div>
                            </>
                        ) : mediaBlob ? (
                            <video ref={recordedVideoRef} src={URL.createObjectURL(mediaBlob)} controls className="w-full h-full"></video>
                        ) : (
                            <div className="w-full h-full bg-gray-100 flex items-center justify-center">
                                <div className="text-center space-y-4">
                                    <div className="w-24 h-24 mx-auto bg-gray-200 rounded-full flex items-center justify-center">
                                        <svg className="w-12 h-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                                        </svg>
                                    </div>
                                    <p className="text-gray-500 font-medium">Your camera preview will appear here</p>
                                </div>
                            </div>
                        )}
                    </div>

                    <div className="flex flex-col items-center space-y-6">
                        <button 
                            onClick={isRecording ? handleStopRecording : handleStartRecording}
                            className={`flex items-center justify-center w-20 h-20 rounded-full text-white font-semibold transition-all duration-200 focus:outline-none shadow-lg hover:shadow-xl ${
                                isRecording 
                                    ? 'bg-red-600 hover:bg-red-700' 
                                    : 'bg-blue-600 hover:bg-blue-700'
                            }`}
                        >
                            <MicIcon className="w-10 h-10" />
                        </button>

                        {!isLoading && !analysisResult && statusText && (
                            <div className="text-center space-y-2">
                                <p className="text-lg text-gray-700 font-medium max-w-md">{statusText}</p>
                            </div>
                        )}

                        {mediaBlob && !isRecording && (
                            <div className="w-full max-w-md">
                                <button 
                                    onClick={handleAnalyze} 
                                    disabled={isLoading} 
                                    className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-medium py-4 px-8 rounded-lg transition-colors duration-200 shadow-sm"
                                >
                                    {isLoading ? (
                                        <div className="flex items-center justify-center">
                                            <LoaderIcon />
                                            <span>Analyzing Performance...</span>
                                        </div>
                                    ) : (
                                        'Analyze My Performance'
                                    )}
                                </button>
                            </div>
                        )}

                        {isLoading && (
                            <div className="text-center space-y-4">
                                <div className="w-8 h-8 border-4 border-gray-300 border-t-blue-600 rounded-full animate-spin mx-auto"></div>
                                <p className="text-blue-700 font-medium">{statusText}</p>
                            </div>
                        )}
                    </div>
                </main>

                {analysisResult && (
                    <div className="w-full">
                        <AnalysisReport data={analysisResult} onGeneratePdf={handleGeneratePdf} />
                    </div>
                )}

                <footer className="text-center space-y-4 pt-12 border-t border-gray-200">
                    <p className="text-gray-500">
                        Powered by FastAPI, React & Computer Vision
                    </p>
                </footer>
            </div>
        </div>
    );
}

export default App;