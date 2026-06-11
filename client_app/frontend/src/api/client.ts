import axios from 'axios';

const apiClient = axios.create({
  baseURL: '/api',
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

// ── Global response interceptor ───────────────────────────────────
// Per AI_RULES.md §4: format errors as [METHOD] [URL] | Status | Payload | Backend detail
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const method = error.config?.method?.toUpperCase() ?? 'UNKNOWN';
    const url = error.config?.url ?? 'UNKNOWN';
    const status = error.response?.status ?? 0;
    const payload = error.config?.data ?? '(no payload)';
    const backendDetail =
      error.response?.data?.error_msg ??
      error.response?.data?.detail ??
      error.message ??
      '(no backend detail)';

    console.error(
      `[${method}] [${url}] | ${status} | Payload: ${payload} | Backend: ${backendDetail}`,
    );

    return Promise.reject(error);
  },
);

export default apiClient;
