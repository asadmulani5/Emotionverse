import { useEffect, useRef } from 'react';

export default function WebcamCapture({ onFrame }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then((stream) => {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      })
      .catch((err) => console.error('Camera error:', err));

    const interval = setInterval(() => {
      if (!videoRef.current || !canvasRef.current) return;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      canvas.width = 640;
      canvas.height = 480;
      ctx.drawImage(videoRef.current, 0, 0, 640, 480);
      const b64 = canvas.toDataURL('image/jpeg', 0.8).split(',')[1];
      onFrame(b64);
    }, 1000);

    return () => clearInterval(interval);
  }, [onFrame]);

  return (
    <div style={{ textAlign: 'center' }}>
      <p style={{ color: '#9ca3af', marginBottom: '0.5rem' }}>Live Camera</p>
      <video
        ref={videoRef}
        style={{ width: 320, height: 240, borderRadius: 8, background: '#1f1f1f' }}
        muted
      />
      <canvas ref={canvasRef} style={{ display: 'none' }} />
    </div>
  );
}