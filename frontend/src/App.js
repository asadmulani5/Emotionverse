import { useEffect, useRef, useState } from 'react';
import { io } from 'socket.io-client';
import EmotionDashboard from './components/EmotionDashboard';
import WebcamCapture from './components/WebcamCapture';
import './App.css';

const BACKEND = 'http://localhost:8000';

export default function App() {
  const socketRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const [emotions, setEmotions] = useState(null);

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
      setEmotions(data);
    });

    return () => socketRef.current.disconnect();
  }, []);

  const sendFrame = (imageB64) => {
    if (socketRef.current) {
      socketRef.current.emit('analyze', { image: imageB64, text: '', audio: [] });
    }
  };

  return (
    <div style={{ minHeight: '100vh', background: '#0f0f0f', color: '#fff', padding: '2rem' }}>
      <h1 style={{ textAlign: 'center', marginBottom: '0.5rem' }}>EmotionVerse</h1>
      <p style={{ textAlign: 'center', color: connected ? '#4ade80' : '#f87171', marginBottom: '2rem' }}>
        {connected ? 'Connected' : 'Connecting...'}
      </p>
      <div style={{ display: 'flex', gap: '2rem', justifyContent: 'center', flexWrap: 'wrap' }}>
        <WebcamCapture onFrame={sendFrame} />
        <EmotionDashboard emotions={emotions} />
      </div>
    </div>
  );
}