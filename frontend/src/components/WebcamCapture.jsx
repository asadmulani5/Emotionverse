import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react';

const FRAME_W = 480;
const FRAME_H = 360;

export function cameraErrorMessage(err) {
  switch (err && err.name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'Camera access was denied. Allow camera permission in your browser settings, or turn the face toggle off.';
    case 'NotFoundError':
    case 'OverconstrainedError':
      return 'No camera was found on this device.';
    case 'NotReadableError':
      return 'The camera is in use by another application.';
    default:
      return 'Could not start the camera.';
  }
}

/** Shows the camera preview when `enabled`; releases the camera when disabled. */
const WebcamCapture = forwardRef(function WebcamCapture({ enabled, onError, onReady }, ref) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!enabled) return undefined;
    let stream;
    let cancelled = false;
    const video = videoRef.current;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      onError?.('This browser does not support camera access (HTTPS or localhost is required).');
      return undefined;
    }
    navigator.mediaDevices
      .getUserMedia({ video: { width: 640, height: 480 } })
      .then((s) => {
        if (cancelled) {
          s.getTracks().forEach((t) => t.stop());
          return;
        }
        stream = s;
        if (video) {
          video.srcObject = s;
          video.play().catch(() => {});
        }
        onReady?.();
      })
      .catch((err) => onError?.(cameraErrorMessage(err), err));
    return () => {
      cancelled = true;
      if (stream) stream.getTracks().forEach((t) => t.stop());
      if (video) video.srcObject = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled]);

  useImperativeHandle(ref, () => ({
    // JPEG base64 of the current frame, or null if the camera isn't producing frames yet
    getFrame() {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!enabled || !video || !canvas || !video.videoWidth) return null;
      canvas.width = FRAME_W;
      canvas.height = FRAME_H;
      canvas.getContext('2d').drawImage(video, 0, 0, FRAME_W, FRAME_H);
      return canvas.toDataURL('image/jpeg', 0.7).split(',')[1];
    },
  }));

  return (
    <div className="video-wrap">
      {enabled ? (
        <video ref={videoRef} muted playsInline aria-label="Live camera preview" />
      ) : (
        <div className="placeholder">Camera is off</div>
      )}
      <canvas ref={canvasRef} style={{ display: 'none' }} />
    </div>
  );
});

export default WebcamCapture;
