import { useCallback, useEffect, useRef, useState } from 'react';
import { io } from 'socket.io-client';
import { BACKEND, RESULT_TIMEOUT_MS } from '../config';

/**
 * Socket.IO connection with auto-reconnect.
 * status: 'connecting' | 'connected' | 'reconnecting' | 'unreachable'
 *  - unreachable: never managed to connect (server down); still retrying.
 *  - reconnecting: connection was lost after being established.
 * send() is stop-and-wait: it refuses while a previous request is still in flight.
 */
export default function useEmotionSocket({ onResult, onServerError } = {}) {
  const [status, setStatus] = useState('connecting');
  const [sessionId, setSessionId] = useState(null);
  const socketRef = useRef(null);
  const inFlight = useRef(false);
  const timer = useRef(null);
  const everConnected = useRef(false);
  const handlers = useRef({});
  handlers.current = { onResult, onServerError };

  useEffect(() => {
    const socket = io(BACKEND, {
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      timeout: 5000,
    });
    socketRef.current = socket;

    const release = () => {
      inFlight.current = false;
      clearTimeout(timer.current);
    };

    socket.on('connect', () => {
      everConnected.current = true;
      release();
      setStatus('connected');
    });
    socket.on('disconnect', (reason) => {
      release();
      if (reason !== 'io client disconnect') setStatus('reconnecting');
    });
    socket.on('connect_error', () => {
      release();
      setStatus(everConnected.current ? 'reconnecting' : 'unreachable');
    });
    socket.on('server_message', (msg) => {
      if (msg && msg.session_id) setSessionId(msg.session_id);
    });
    socket.on('emotion_result', (data) => {
      release();
      if (!data || data.dropped) return; // server shed this frame; try the next one
      if (data.error) {
        handlers.current.onServerError?.(data);
        return;
      }
      handlers.current.onResult?.(data);
    });

    return () => {
      release();
      socket.disconnect();
    };
  }, []);

  const send = useCallback((payload) => {
    const socket = socketRef.current;
    if (!socket || !socket.connected || inFlight.current) return false;
    inFlight.current = true;
    timer.current = setTimeout(() => {
      inFlight.current = false;
    }, RESULT_TIMEOUT_MS);
    socket.emit('analyze', payload);
    return true;
  }, []);

  return { status, sessionId, send };
}
