// messages: [{ id, level: 'error'|'warn'|'info', text }]
export default function ErrorBanner({ messages }) {
  if (!messages.length) return null;
  return (
    <div className="banners" role="alert">
      {messages.map((m) => (
        <div key={m.id} className={`banner ${m.level}`}>{m.text}</div>
      ))}
    </div>
  );
}
