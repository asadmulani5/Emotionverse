import { useEffect, useRef, useState } from 'react';
import { io } from 'socket.io-client';
import EmotionDashboard from './components/EmotionDashboard';
import WebcamCapture from './components/WebcamCapture';
import './App.css';

const BACKEND = 'http://localhost:8000';

export default function App() {
  const socketRef = useRef(null);

  const [connected, setConnected] = useState(false);
  const [userInput, setUserInput] = useState("");
  const [history, setHistory] = useState([]);
  const [mode, setMode] = useState("fusion");
  const [log, setLog] = useState([]); // ✅ TIMELINE

  useEffect(() => {
    socketRef.current = io(BACKEND);

    socketRef.current.on('connect', () => {
      setConnected(true);
      console.log('Connected to EmotionVerse backend');
    });

    socketRef.current.on('disconnect', () => {
      setConnected(false);
    });

    socketRef.current.on('emotion_result', (data) => {
      let newEmotion;

      if (mode === "text") newEmotion = data.text;
      else if (mode === "face") newEmotion = data.face;
      else newEmotion = data.fused;

      if (!newEmotion || !newEmotion.emotions) return;

      // ✅ SMOOTHING BUFFER
      setHistory(prev => {
        const updated = [...prev, newEmotion].slice(-5);
        return updated;
      });

      // ✅ TIMELINE LOG
      setLog(prev => [
        ...prev,
        {
          time: new Date().toLocaleTimeString(),
          emotion: newEmotion.dominant
        }
      ].slice(-20));
    });

    return () => socketRef.current.disconnect();
  }, [mode]); // ✅ IMPORTANT FIX

  const sendFrame = (imageB64) => {
    if (socketRef.current) {
      socketRef.current.emit('analyze', {
        image: imageB64,
        text: userInput,
        audio: []
      });
    }
  };

  // ✅ SMOOTHING FUNCTION
  const getSmoothedEmotion = () => {
    if (history.length === 0) return null;

    const avg = {};

    history.forEach(e => {
      Object.entries(e.emotions).forEach(([k, v]) => {
        avg[k] = (avg[k] || 0) + v;
      });
    });

    Object.keys(avg).forEach(k => avg[k] /= history.length);

    const dominant = Object.keys(avg).reduce((a, b) =>
      avg[a] > avg[b] ? a : b
    );

    return { emotions: avg, dominant };
  };

  const smoothed = getSmoothedEmotion();

  return (
    <div style={{ minHeight: '100vh', background: '#0f0f0f', color: '#fff', padding: '2rem' }}>
      
      <h1 style={{ textAlign: 'center', marginBottom: '0.5rem' }}>
        EmotionVerse
      </h1>

      {/* MODE */}
      <p style={{ textAlign: 'center', marginBottom: '1rem' }}>
        Mode: {mode.toUpperCase()}
      </p>

      <div style={{ display: 'flex', justifyContent: 'center', gap: '10px', marginBottom: '1rem' }}>
      <button style={{ padding: '8px 16px', borderRadius: '8px', background: '#1f2937', color: '#fff', border: 'none' }} onClick={() => setMode("text")}>Text</button>
<button style={{ padding: '8px 16px', borderRadius: '8px', background: '#1f2937', color: '#fff', border: 'none' }} onClick={() => setMode("face")}>Face</button>
<button style={{ padding: '8px 16px', borderRadius: '8px', background: '#2563eb', color: '#fff', border: 'none' }} onClick={() => setMode("fusion")}>Fusion</button>
      </div>

      <p style={{ textAlign: 'center', color: connected ? '#4ade80' : '#f87171', marginBottom: '2rem' }}>
        {connected ? 'Connected' : 'Connecting...'}
      </p>

      {/* INPUT */}
      <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '2rem' }}>
        <input
          type="text"
          placeholder="Type how you feel..."
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          style={{
            width: '400px',
            padding: '12px',
            borderRadius: '10px',
            border: 'none',
            outline: 'none',
            fontSize: '16px'
          }}
        />
      </div>

      <div style={{ display: 'flex', gap: '2rem', justifyContent: 'center', flexWrap: 'wrap' }}>
        <WebcamCapture onFrame={sendFrame} />
        <EmotionDashboard emotions={smoothed} />
      </div>

      {/* DOMINANT */}
      {smoothed && (
        <h2 style={{ textAlign: 'center', marginTop: '2rem' }}>
          <span style={{
  color:
    smoothed.dominant === "happy" ? "#4ade80" :
    smoothed.dominant === "sad" ? "#60a5fa" :
    smoothed.dominant === "angry" ? "#f87171" :
    "#9ca3af"
}}>
  {smoothed.dominant}
</span> (
          {(smoothed.emotions[smoothed.dominant] * 100).toFixed(1)}%)
        </h2>
      )}

      {/* TIMELINE */}
      {log.length > 0 && (
        <div style={{ marginTop: '2rem', textAlign: 'center' }}>
          <h3>Emotion Timeline</h3>
          <ul style={{ listStyle: 'none', padding: 0 }}>
            {log.map((item, i) => (
              <li key={i}>
                {item.time} → {item.emotion}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}