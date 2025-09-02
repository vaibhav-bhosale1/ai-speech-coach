import React, { useState, useRef, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, Radar, PieChart, Pie, Cell, Legend } from 'recharts';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

// --- SVG Icons ---
const MicIcon = ({ className }) => (<svg className={className} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"></path><path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"></path></svg>);
const LoaderIcon = () => (<svg className="animate-spin h-5 w-5 mr-3" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2.99988V5.99988" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/><path d="M12 18.0001V21.0001" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/><path d="M5.63574 5.63623L7.75674 7.75723" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/><path d="M16.2422 16.2427L18.3632 18.3637" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/><path d="M3 12.0001H6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/><path d="M18 12.0001H21" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/><path d="M5.63574 18.3637L7.75674 16.2427" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/><path d="M16.2422 7.75723L18.3632 5.63623" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>);
const ClipboardIcon = ({ className }) => (<svg className={className} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path d="M7 3a1 1 0 000 2h6a1 1 0 100-2H7zM4 7a1 1 0 011-1h10a1 1 0 110 2H5a1 1 0 01-1-1zM2 11a2 2 0 012-2h12a2 2 0 012 2v3a2 2 0 01-2 2H4a2 2 0 01-2-2v-3z"></path></svg>);
const CheckIcon = ({ className }) => (<svg className={className} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd"></path></svg>);


// --- Reusable UI Components ---
const StatCard = ({ title, value, subtext, colorClass = 'text-white' }) => (
    <div className="bg-gray-700/50 p-4 rounded-lg text-center shadow-lg flex flex-col justify-center transition-all duration-300 hover:bg-gray-700/80">
        <h3 className="text-xs md:text-sm text-gray-400 font-medium uppercase tracking-wider">{title}</h3>
        <p className={`text-2xl md:text-3xl font-bold ${colorClass}`}>{value}</p>
        {subtext && !isNaN(parseFloat(value)) && <p className="text-xs text-gray-500">{subtext}</p>}
    </div>
);
const ChartPlaceholder = ({ message }) => (
    <div className="bg-gray-700/50 p-4 rounded-lg flex items-center justify-center h-full min-h-[250px]">
        <p className="text-gray-400 text-center">{message}</p>
    </div>
);
const ReportSection = ({ title, children }) => (
    <div className="bg-gray-700/50 p-4 rounded-lg shadow-lg">
        <h3 className="text-md font-semibold text-gray-300 mb-4">{title}</h3>
        {children}
    </div>
);

// --- Chart Components ---
const EmotionChart = ({ data }) => {
    const COLORS = { Happy: '#10B981', Sad: '#3B82F6', Angry: '#EF4444', Neutral: '#6B7280', Surprise: '#F59E0B', Fear: '#8B5CF6', Disgust: '#D946EF' };
    const chartData = data ? Object.entries(data).map(([name, value]) => ({ name, value })) : [];
    if (chartData.length === 0) return <ChartPlaceholder message="No facial emotion data detected." />;
    return (
        <ResponsiveContainer width="100%" height={250}>
            <PieChart>
                <Pie data={chartData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} labelLine={false}>
                    {chartData.map((entry, index) => (<Cell key={`cell-${index}`} fill={COLORS[entry.name] || '#8884d8'} />))}
                </Pie>
                <Tooltip contentStyle={{backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '0.5rem'}}/>
                <Legend />
            </PieChart>
        </ResponsiveContainer>
    );
};
const FillerWordsChart = ({ data }) => {
    const chartData = data ? Object.entries(data).map(([name, count]) => ({ name, count })) : [];
    if (chartData.length === 0) return null; 
    return (
        <ResponsiveContainer width="100%" height={250}>
            <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 20, left: 20, bottom: 5 }}>
                <XAxis type="number" hide />
                <YAxis type="category" dataKey="name" width={60} stroke="#9ca3af" tick={{ fill: '#d1d5db' }} interval={0}/>
                <Tooltip cursor={{fill: 'rgba(255, 255, 255, 0.1)'}} contentStyle={{backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '0.5rem'}}/>
                <Bar dataKey="count" fill="#4f46e5" barSize={20} />
            </BarChart>
        </ResponsiveContainer>
    );
};
const AudioQualityChart = ({ data }) => {
    if(data?.pitch_variation === 'N/A') return <ChartPlaceholder message="Audio was silent or too quiet for quality analysis."/>
    const chartData = [{ subject: 'Confidence', A: data?.confidence_score || 0, fullMark: 100 },{ subject: 'Volume', A: data?.volume_consistency || 0, fullMark: 100 },{ subject: 'Pitch', A: data?.pitch_variation === 'Dynamic' ? 85 : 40, fullMark: 100 },];
    return (
        <ResponsiveContainer width="100%" height={250}>
            <RadarChart cx="50%" cy="50%" outerRadius="80%" data={chartData}>
                <PolarGrid stroke="#4b5563" />
                <PolarAngleAxis dataKey="subject" stroke="#9ca3af" tick={{ fill: '#d1d5db' }} />
                <Radar name="Score" dataKey="A" stroke="#4f46e5" fill="#4f46e5" fillOpacity={0.6} />
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
  
  const getWpmColor = (wpm) => (wpm > 0 && wpm < 130) ? 'text-yellow-400' : wpm > 160 ? 'text-red-400' : 'text-green-400';
  const getPostureColor = (score) => (score < 80) ? 'text-red-400' : score < 95 ? 'text-yellow-400' : 'text-green-400';
  const getTiltColor = (status) => (status !== "Good") ? 'text-yellow-400' : 'text-green-400';
 
  return (
    <div id="analysisReport" className="w-full max-w-5xl mx-auto bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl shadow-2xl p-6 space-y-6">
      <div className="flex justify-between items-center border-b border-gray-700 pb-4">
        <h2 className="text-2xl font-bold">Analysis Report</h2>
        <button onClick={onGeneratePdf} className="bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded-lg transition-colors text-sm">Download PDF</button>
      </div>
      
      <ReportSection title="Key Metrics">
        <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
            <StatCard title="Speaking Pace" value={noSpeechDetected ? 'N/A' : (data.wpm || 0)} subtext="WPM" colorClass={getWpmColor(data.wpm)} />
            <StatCard title="Filler Words" value={noSpeechDetected ? 'N/A' : fillerWordCount} colorClass={fillerWordCount > 5 ? 'text-red-400' : 'text-green-400'} />
            <StatCard title="Sentiment" value={noSpeechDetected ? 'N/A' : (data.sentiment?.label || 'N/A')} />
            <StatCard title="Eye Contact" value={data.eye_contact?.gaze_stability ?? 'N/A'} subtext="% stability" />
            <StatCard title="Posture Score" value={data.posture?.score ?? 'N/A'} subtext="/ 100" colorClass={getPostureColor(data.posture?.score)} />
            <StatCard title="Head Tilt" value={data.posture?.tilt_status ?? 'N/A'} colorClass={getTiltColor(data.posture?.tilt_status)} />
        </div>
      </ReportSection>

      <ReportSection title="Transcription">
        <div className="relative">
            <button onClick={handleCopy} className="absolute top-[-2.5rem] right-0 bg-gray-600 hover:bg-gray-500 text-white text-xs px-2 py-1 rounded transition-colors">
                {copied ? <span className="flex items-center"><CheckIcon className="w-4 h-4 mr-1"/> Copied</span> : <span className="flex items-center"><ClipboardIcon className="w-4 h-4 mr-1"/> Copy</span>}
            </button>
            <p className="bg-gray-800 p-4 rounded-md text-gray-300 whitespace-pre-wrap font-mono text-sm leading-relaxed max-h-40 overflow-y-auto">{data.transcription || ""}</p>
        </div>
      </ReportSection>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <ReportSection title="Facial Emotion Profile">
          <EmotionChart data={data.video_analysis?.emotion_distribution} />
        </ReportSection>
        <ReportSection title="Audio Quality Profile">
            {noSpeechDetected ? (<ChartPlaceholder message="Audio metrics are unavailable as no speech was detected." />) 
                             : (<AudioQualityChart data={data.audio_quality} />)
            }
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
    const [statusText, setStatusText] = useState('Press the button to start recording');
    
    const mediaRecorderRef = useRef(null);
    const videoRef = useRef(null);
    const recordedVideoRef = useRef(null);

    useEffect(() => { if (stream && videoRef.current) { videoRef.current.srcObject = stream; } }, [stream]);

    // CORRECTED: All handler functions are now defined INSIDE the App component
    const handleGeneratePdf = async () => {
        const reportElement = document.getElementById('analysisReport');
        if (!reportElement) return;
        setStatusText('Generating PDF...');
        const originalBg = reportElement.style.backgroundColor;
        reportElement.style.backgroundColor = '#111827';
        const canvas = await html2canvas(reportElement, { scale: 2, useCORS: true, backgroundColor: '#111827' });
        reportElement.style.backgroundColor = originalBg;
        const imgData = canvas.toDataURL('image/png');
        const pdf = new jsPDF({ orientation: 'p', unit: 'px', format: [canvas.width, canvas.height] });
        pdf.addImage(imgData, 'PNG', 0, 0, canvas.width, canvas.height);
        pdf.save(`Speech-Analysis-Report-${new Date().toISOString().slice(0,10)}.pdf`);
        setStatusText('PDF downloaded.');
    };

    const handleStartRecording = async () => {
        try {
            const mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: true });
            setStream(mediaStream);
            setIsRecording(true);
            setMediaBlob(null);
            setAnalysisResult(null);
            setStatusText('Recording in progress...');
            const chunks = [];
            mediaRecorderRef.current = new MediaRecorder(mediaStream, { mimeType: 'video/webm' });
            mediaRecorderRef.current.ondataavailable = event => chunks.push(event.data);
            mediaRecorderRef.current.onstop = () => {
                const blob = new Blob(chunks, { type: 'video/webm' });
                setMediaBlob(blob);
                mediaStream.getTracks().forEach(track => track.stop());
                setStream(null);
                setStatusText('Recording finished. Ready to analyze.');
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
        setStatusText('Hold on, the AI is analyzing your performance...');
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
            setStatusText('Analysis complete. View your report below.');
        }
    };

    return (
        <div className="bg-gray-900 text-white min-h-screen flex flex-col items-center p-4 font-sans selection:bg-indigo-500 selection:text-white">
            <div className="w-full max-w-5xl space-y-8 py-8">
                <header className="text-center">
                    <h1 className="text-4xl md:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 to-purple-500">AI Speech & Video Coach</h1>
                    <p className="text-gray-400 mt-2">Get instant, data-driven feedback on your communication skills.</p>
                </header>
                <main className="flex flex-col items-center space-y-6">
                    <div className="w-full max-w-2xl bg-black rounded-lg overflow-hidden aspect-video shadow-2xl border border-gray-700">
                        {isRecording ? (<video ref={videoRef} autoPlay muted className="w-full h-full object-cover"></video>) 
                        : mediaBlob ? (<video ref={recordedVideoRef} src={URL.createObjectURL(mediaBlob)} controls className="w-full h-full"></video>) 
                        : (<div className="w-full h-full bg-gray-800 flex items-center justify-center"><p className="text-gray-500">Your camera feed will appear here</p></div>)}
                    </div>
                    <button onClick={isRecording ? handleStopRecording : handleStartRecording} className={`flex items-center justify-center w-24 h-24 rounded-full text-white font-semibold transition-all duration-300 ease-in-out focus:outline-none focus:ring-4 shadow-lg ${isRecording ? 'bg-red-600 hover:bg-red-700 focus:ring-red-500 animate-pulse' : 'bg-indigo-600 hover:bg-indigo-700 focus:ring-indigo-500'}`}>
                        <MicIcon className="w-10 h-10" />
                    </button>
                    <p className="text-lg h-6 text-gray-400">{!isLoading && !analysisResult ? statusText : ''}</p>
                    {mediaBlob && !isRecording && (
                        <div className="w-full max-w-md bg-gray-800/50 backdrop-blur-sm border border-gray-700 p-4 rounded-lg shadow-xl">
                            <button onClick={handleAnalyze} disabled={isLoading} className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-500 disabled:cursor-not-allowed flex items-center justify-center text-white font-bold py-3 rounded-lg transition-colors">
                                {isLoading ? <><LoaderIcon /> Analyzing...</> : 'Analyze My Performance'}
                            </button>
                        </div>
                    )}
                    {isLoading && <p className="text-center text-indigo-400">{statusText}</p>}
                </main>
                {analysisResult && <AnalysisReport data={analysisResult} onGeneratePdf={handleGeneratePdf} />}
                <footer className="text-center text-gray-500 text-sm pt-8">
                    <p>Powered by FastAPI, React & Computer Vision</p>
                </footer>
            </div>
        </div>
    );
}

export default App;