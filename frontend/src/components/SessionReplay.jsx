import { useEffect, useMemo, useState } from 'react';
import ExpressionPanel from './ExpressionPanel';
import { describePair } from './MismatchAlert';
import { formatDuration } from '../utils/format';

const SPEEDS = [1, 2, 4];
const MAX_GAP_MS = 2000; // long pauses in the recording are shortened during replay

export default function SessionReplay({ results }) {
  const frames = useMemo(() => (results || []).filter((r) => r.fused && r.fused.dominant), [results]);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);

  useEffect(() => {
    setIndex(0);
    setPlaying(false);
  }, [frames]);

  useEffect(() => {
    if (!playing) return undefined;
    if (index >= frames.length - 1) {
      setPlaying(false);
      return undefined;
    }
    const gap = new Date(frames[index + 1].timestamp) - new Date(frames[index].timestamp);
    const delay = Math.min(Math.max(gap, 100), MAX_GAP_MS) / speed;
    const id = setTimeout(() => setIndex((i) => i + 1), delay);
    return () => clearTimeout(id);
  }, [playing, index, speed, frames]);

  if (!frames.length) {
    return (
      <section className="card full" aria-label="Replay">
        <h2>Replay</h2>
        <p className="muted small-text">Nothing to replay.</p>
      </section>
    );
  }

  const frame = frames[index];
  const offset = (new Date(frame.timestamp) - new Date(frames[0].timestamp)) / 1000;
  const mismatch = frame.mismatch && frame.mismatch.detected
    ? frame.mismatch.pairs.map(describePair).join('; ')
    : null;
  const atEnd = index >= frames.length - 1;

  return (
    <>
      <section className="card" aria-label="Replay controls">
        <h2>Replay</h2>
        <div className="player">
          <button
            className="btn primary"
            onClick={() => {
              if (atEnd) setIndex(0);
              setPlaying((p) => !p);
            }}
          >
            {playing ? 'Pause' : atEnd ? 'Replay' : 'Play'}
          </button>
          <input
            type="range"
            min={0}
            max={frames.length - 1}
            value={index}
            aria-label="Position in session"
            onChange={(e) => {
              setPlaying(false);
              setIndex(Number(e.target.value));
            }}
          />
          <select className="btn small" value={speed} aria-label="Playback speed" onChange={(e) => setSpeed(Number(e.target.value))}>
            {SPEEDS.map((s) => <option key={s} value={s}>{s}x</option>)}
          </select>
        </div>
        <p className="muted small-text" style={{ margin: '8px 0 0' }}>
          Reading {index + 1} of {frames.length} · +{formatDuration(offset)} · inputs: {(frame.fused.used || []).join(', ') || '—'}
        </p>
        {mismatch && (
          <div className="banner warn small-text" style={{ marginTop: 10 }}>Signals disagreed: {mismatch}</div>
        )}
      </section>
      <ExpressionPanel smoothed={{ emotions: frame.fused.emotions, dominant: frame.fused.dominant }} />
    </>
  );
}
