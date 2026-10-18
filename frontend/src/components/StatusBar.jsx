const LABELS = {
  connecting: 'Connecting…',
  connected: 'Connected',
  reconnecting: 'Reconnecting…',
  unreachable: 'Server unreachable, retrying…',
};

export default function StatusBar({ status }) {
  return (
    <span className={`status ${status}`} role="status" aria-live="polite">
      <span className="dot" aria-hidden="true" />
      {LABELS[status] || status}
    </span>
  );
}
