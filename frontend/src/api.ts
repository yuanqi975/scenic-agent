// Same-origin by default; Vite proxies /api to the backend during development.
// This avoids browser-specific localhost/CORS failures when opening the UI via
// 127.0.0.1, a LAN address, or a reverse proxy.
const BASE = import.meta.env.VITE_API_BASE || '/api/v1'
export async function api(path: string, init?: RequestInit) {
  const response = await fetch(`${BASE}${path}`, { credentials: 'include', headers: {'Content-Type': 'application/json', ...(init?.headers || {})}, ...init })
  if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`)
  return response.json()
}
export async function streamChat(message: string, onEvent: (event: string, data: any) => void, conversationId?: string) {
  const response = await fetch(`${BASE}/chat/stream`, { method:'POST', credentials:'include', headers:{'Content-Type':'application/json'}, body: JSON.stringify({message, conversation_id: conversationId || null}) })
  if (!response.body) throw new Error('未收到流式响应')
  const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = ''
  while (true) { const {value, done} = await reader.read(); if (done) break; buffer += decoder.decode(value, {stream:true}); const chunks = buffer.split('\n\n'); buffer = chunks.pop() || ''; for (const chunk of chunks) { const event = chunk.match(/^event: (.+)$/m)?.[1]; const data = chunk.match(/^data: (.+)$/m)?.[1]; if (event && data) onEvent(event, JSON.parse(data)) } }
}
