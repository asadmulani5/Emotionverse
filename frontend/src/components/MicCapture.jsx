import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';
import { float32ToBase64, resample, rms } from '../utils/audio';

const KEEP_SECONDS = 3;
const MIN_SECONDS = 1;
const SILENCE_RMS = 0.01; // below this we don't send: the voice model would just guess on silence
const TARGET_RATE = 16000;

export function micErrorMessage(err) {
  switch (err && err.name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'Microphone access was denied. Allow microphone permission in your browser settings, or turn the voice toggle off.';
    case 'NotFoundError':
      return 'No microphone was found on this device.';
    case 'NotReadableError':
      return 'The microphone is in use by another application.';
    default:
      return 'Could not start the microphone.';
  }
}

/** Invisible component: records a rolling window of mic audio while `enabled`; getAudio() returns the latest clip. */
const MicCapture = forwardRef(function MicCapture({ enabled, onError, onReady }, ref) {
  const chunks = useRef([]);
  const rate = useRef(TARGET_RATE);

  useEffect(() => {
    if (!enabled) return undefined;
    let stream;
    let ctx;
    let node;
    let source;
    let cancelled = false;
    chunks.current = [];

    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !Ctx) {
      onError?.('This browser does not support microphone access (HTTPS or localhost is required).');
      return undefined;
    }
    navigator.mediaDevices
      .getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } })
      .then((s) => {
        if (cancelled) {
          s.getTracks().forEach((t) => t.stop());
          return;
        }
        stream = s;
        ctx = new Ctx();
        rate.current = ctx.sampleRate;
        source = ctx.createMediaStreamSource(s);
        node = ctx.createScriptProcessor(4096, 1, 1);
        node.onaudioprocess = (e) => {
          chunks.current.push(new Float32Array(e.inputBuffer.getChannelData(0)));
          const maxChunks = Math.ceil((KEEP_SECONDS * rate.current) / 4096);
          if (chunks.current.length > maxChunks) chunks.current.shift();
        };
        source.connect(node);
        node.connect(ctx.destination); // required for onaudioprocess to fire; output is silent
        onReady?.();
      })
      .catch((err) => onError?.(micErrorMessage(err), err));

    return () => {
      cancelled = true;
      if (node) node.onaudioprocess = null;
      if (source) source.disconnect();
      if (node) node.disconnect();
      if (ctx) ctx.close().catch(() => {});
      if (stream) stream.getTracks().forEach((t) => t.stop());
      chunks.current = [];
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled]);

  useImperativeHandle(ref, () => ({
    getAudio() {
      if (!enabled || !chunks.current.length) return null;
      const total = chunks.current.reduce((n, c) => n + c.length, 0);
      if (total < MIN_SECONDS * rate.current) return null;
      const all = new Float32Array(total);
      let off = 0;
      chunks.current.forEach((c) => {
        all.set(c, off);
        off += c.length;
      });
      if (rms(all) < SILENCE_RMS) return null;
      return { audio: float32ToBase64(resample(all, rate.current, TARGET_RATE)), sample_rate: TARGET_RATE };
    },
  }));

  return null;
});

export default MicCapture;
