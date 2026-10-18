import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ErrorBanner from './components/ErrorBanner';
import ExpressionPanel from './components/ExpressionPanel';
import StatusBar from './components/StatusBar';
import Timeline from './components/Timeline';
import WebcamCapture from './components/WebcamCapture';
import { BACKEND, MIN_SEND_INTERVAL_MS, SMOOTHING_WINDOW } from './config';
import useEmotionSocket from './hooks/useEmotionSocket';
import { smooth } from './utils/smoothing';

let nextId = 1;

export default function App() {
  const camRef = useRef(null);
  const [text, setText] = useState('');
  const [submittedText, setSubmittedText] = useState('');
  const [history, setHistory] = useState([]);
  const [timeline, setTimeline] = useState([]);
  const [cameraError, setCameraError] = useState(null);
  const [faceIssue, setFaceIssue] = useState(null);
  const [serverError, setServerError] = useState(null);
  const lastDominant = useRef(null);

  const onResult = useCallback((data) => {
    setServerError(null);
    if (data.face && data.face.error) {
      setFaceIssue(/no face/i.test(data.face.error)
        ? 'No face detected. Check the lighting and that your face is in view.'
        : null);
    } else if (data.face && data.face.emotions) {
      setFaceIssue(null);
    }
    const fused = data.fused;
    if (!fused || !fused.emotions || !fused.dominant) return;
    setHistory((h) => [...h, fused.emotions].slice(-SMOOTHING_WINDOW));
    if (fused.dominant !== lastDominant.current) {
      lastDominant.current = fused.dominant;
      setTimeline((t) => [...t, { id: nextId++, time: new Date().toLocaleTimeString(), emotion: fused.dominant }].slice(-30));
    }
  }, []);

  const onServerError = useCallback((data) => setServerError(data.error), []);
  const { status, send } = useEmotionSocket({ onResult, onServerError });

  // Send loop: at most once per interval, and only when the previous request has returned.
  useEffect(() => {
    const id = setInterval(() => {
      const payload = {};
      const frame = camRef.current && camRef.current.getFrame();
      if (frame) payload.image = frame;
      if (submittedText) payload.text = submittedText;
      if (Object.keys(payload).length) send(payload);
    }, MIN_SEND_INTERVAL_MS);
    return () => clearInterval(id);
  }, [send, submittedText]);

  const smoothed = useMemo(() => smooth(history), [history]);

  const banners = [];
  if (status === 'unreachable') banners.push({ id: 'srv', level: 'error', text: `Can't reach the EmotionVerse server at ${BACKEND}. Retrying automatically…` });
  if (status === 'reconnecting') banners.push({ id: 'rec', level: 'warn', text: 'Connection lost. Reconnecting…' });
  if (cameraError) banners.push({ id: 'cam', level: 'error', text: cameraError });
  if (faceIssue) banners.push({ id: 'face', level: 'warn', text: faceIssue });
  if (serverError) banners.push({ id: 'err', level: 'warn', text: `Server could not analyse that input: ${serverError}` });

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <h1>EmotionVerse</h1>
          <p>A private emotion mirror for self-reflection</p>
        </div>
        <StatusBar status={status} />
      </header>

      <ErrorBanner messages={banners} />

      <div className="grid">
        <section className="card" aria-label="Camera">
          <h2>Camera</h2>
          <WebcamCapture ref={camRef} enabled onError={(msg) => setCameraError(msg)} onReady={() => setCameraError(null)} />
        </section>
        <ExpressionPanel smoothed={smoothed} />
        <section className="card full" aria-label="Text input">
          <h2>Text (optional)</h2>
          <form
            className="text-input"
            onSubmit={(e) => {
              e.preventDefault();
              setSubmittedText(text.trim());
            }}
          >
            <input
              type="text"
              value={text}
              maxLength={1000}
              placeholder="Type something to analyse its tone…"
              aria-label="Text to analyse"
              onChange={(e) => setText(e.target.value)}
            />
            <button className="btn primary" type="submit">Analyse</button>
            <button className="btn" type="button" onClick={() => { setText(''); setSubmittedText(''); }}>Clear</button>
          </form>
        </section>
        <Timeline entries={timeline} />
      </div>
      <p className="footer">Detected expressions are estimates, not facts about feelings.</p>
    </div>
  );
}
