import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip);

export default function EmotionDashboard({ emotions }) {
  const defaultEmotions = {
    happy: 0, sad: 0, angry: 0,
    neutral: 1, surprise: 0, fear: 0, disgust: 0
  };

  const data = emotions?.emotions || defaultEmotions;
  const labels = Object.keys(data);
  const values = Object.values(data);

  const chartData = {
    labels,
    datasets: [{
      label: 'Emotion Score',
      data: values,
      backgroundColor: [
        '#4ade80', '#60a5fa', '#f87171',
        '#9ca3af', '#facc15', '#c084fc', '#fb923c'
      ],
      borderRadius: 6,
    }]
  };

  const options = {
    responsive: true,
    scales: {
      y: { min: 0, max: 1, ticks: { color: '#9ca3af' }, grid: { color: '#1f1f1f' } },
      x: { ticks: { color: '#9ca3af' }, grid: { color: '#1f1f1f' } }
    },
    plugins: { legend: { display: false } }
  };

  return (
    <div style={{ width: 400 }}>
      <p style={{ color: '#9ca3af', marginBottom: '0.5rem', textAlign: 'center' }}>
        Live Emotions
      </p>
      {emotions && (
        <p style={{ textAlign: 'center', fontSize: '1.2rem', marginBottom: '1rem' }}>
          Dominant: <strong style={{ color: '#4ade80' }}>{emotions.dominant}</strong>
        </p>
      )}
      <Bar data={chartData} options={options} />
    </div>
  );
}