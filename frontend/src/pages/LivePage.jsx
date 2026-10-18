import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import ErrorBanner from '../components/ErrorBanner';
import ExpressionPanel from '../components/ExpressionPanel';
import MicCapture from '../components/MicCapture';
import ModalityToggles from '../components/ModalityToggles';
import StatusBar from '../components/StatusBar';
import Timeline from '../components/Timeline';
import WebcamCapture from '../components/WebcamCapture';
import { BACKEND, MIN_SEND_INTERVAL_MS, SMOOTHING_WINDOW } from '../config';
import useEmotionSocket from '../hooks/useEmotionSocket';
import { smooth } from '../utils/smoothing';

let nextId = 1;

export default function LivePage() {
  const camRef = useRef(null);
  const micRef = useRef(null);
  const [on, setOn] = useState({ face: true, voice: false, text: true });
  const [micError, setMicError] = useState(null);
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
      const frame = on.face && camRef.current && camRef.current.getFrame();
      if (frame) payload.image = frame;
      const clip = on.voice && micRef.current && micRef.current.getAudio();
      if (clip) Object.assign(payload, clip);
      if (on.text && submittedText) payload.text = submittedText;
      if (Object.keys(payload).length) send(payload);
    }, MIN_SEND_INTERVAL_MS);
    return () => clearInterval(id);
  }, [send, submittedText, on]);

  const smoothed = useMemo(() => smooth(history), [history]);

  const banners = [];
  if (status === 'unreachable') banners.push({ id: 'srv', level: 'error', text: `Can't reach the EmotionVerse server at ${BACKEND}. Retrying automatically…` });
  if (status === 'reconnecting') banners.push({ id: 'rec', level: 'warn', text: 'Connection lost. Reconnecting…' });
  if (on.face && cameraError) banners.push({ id: 'cam', level: 'error', text: cameraError });
  if (on.voice && micError) banners.push({ id: 'mic', level: 'error', text: micError });
  if (on.face && faceIssue) banners.push({ id: 'face', level: 'warn', text: faceIssue });
  if (serverError) banners.push({ id: 'err', level: 'warn', text: `Server could not analyse that input: ${serverError}` });

  return (
    <>
      <div className="status-row"><StatusBar status={status} /></div>
      <ErrorBanner messages={banners} />

      <div className="grid">
        <section className="card full" aria-label="Inputs">
          <h2>Inputs</h2>
          <ModalityToggles value={on} onChange={(next) => { setOn(next); if (!next.voice) setMicError(null); if (!next.face) setCameraError(null); }} />
          <MicCapture ref={micRef} enabled={on.voice} onError={(msg) => setMicError(msg)} onReady={() => setMicError(null)} />
        </section>
        <section className="card" aria-label="Camera">
          <h2>Camera</h2>
          <WebcamCapture ref={camRef} enabled={on.face} onError={(msg) => setCameraError(msg)} onReady={() => setCameraError(null)} />
        </section>
        <ExpressionPanel smoothed={smoothed} />
        <section className="card full" aria-label="Text input">
          <h2>Text (optional){!on.text && ' (off)'}</h2>
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
              disabled={!on.text}
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
    </>
  );
}
