import { useState } from 'react';
import apiClient from './api/client';

interface HealthResponse {
  status: string;
  version: string;
}

function App() {
  const [result, setResult] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const checkHealth = async () => {
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const { data } = await apiClient.get<HealthResponse>('/health');
      setResult(`✅ Backend OK — status: ${data.status}, version: ${data.version}`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unknown error';
      setError(`❌ Connection failed: ${msg}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '2rem', fontFamily: 'monospace' }}>
      <h1>Laser Engraver V2</h1>
      <p>连通性测试</p>

      <button
        onClick={checkHealth}
        disabled={loading}
        style={{
          padding: '0.75rem 1.5rem',
          fontSize: '1rem',
          cursor: loading ? 'wait' : 'pointer',
        }}
      >
        {loading ? '正在测试...' : '测试后端连通性'}
      </button>

      {result && (
        <div style={{ marginTop: '1rem', padding: '1rem', background: '#e6ffe6', borderRadius: '4px' }}>
          {result}
        </div>
      )}

      {error && (
        <div style={{ marginTop: '1rem', padding: '1rem', background: '#ffe6e6', borderRadius: '4px' }}>
          {error}
        </div>
      )}
    </div>
  );
}

export default App;
