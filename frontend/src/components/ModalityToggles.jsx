const ITEMS = [
  { key: 'face', label: 'Face' },
  { key: 'voice', label: 'Voice' },
  { key: 'text', label: 'Text' },
];

/** Each input can be switched off; off means it is neither captured nor sent. */
export default function ModalityToggles({ value, onChange }) {
  return (
    <div className="toggles" role="group" aria-label="Inputs">
      {ITEMS.map(({ key, label }) => (
        <button
          key={key}
          type="button"
          className="toggle"
          aria-pressed={value[key]}
          onClick={() => onChange({ ...value, [key]: !value[key] })}
        >
          {label}: {value[key] ? 'on' : 'off'}
        </button>
      ))}
    </div>
  );
}
