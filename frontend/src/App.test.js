import { fireEvent, render, screen } from '@testing-library/react';
import App from './App';

// The real pages need a webcam, microphone and backend; stub them for app-flow tests.
jest.mock('./pages/LivePage', () => () => 'live page');
jest.mock('./pages/SessionsPage', () => () => 'sessions page');

beforeEach(() => window.localStorage.clear());

test('shows consent first, then the app after accepting', () => {
  render(<App />);
  expect(screen.getByText('Before you start')).toBeInTheDocument();
  expect(screen.queryByText('live page')).not.toBeInTheDocument();

  fireEvent.click(screen.getByText('I understand, continue'));
  expect(screen.getByText('live page')).toBeInTheDocument();

  fireEvent.click(screen.getByText('Sessions'));
  expect(screen.getByText('sessions page')).toBeInTheDocument();
});

test('remembers consent between visits', () => {
  window.localStorage.setItem('ev_consent_v1', 'granted');
  render(<App />);
  expect(screen.getByText('live page')).toBeInTheDocument();
});

test('revoking consent returns to the consent screen', () => {
  window.localStorage.setItem('ev_consent_v1', 'granted');
  render(<App />);
  fireEvent.click(screen.getByText('Privacy & consent'));
  expect(screen.getByText('Before you start')).toBeInTheDocument();
});
