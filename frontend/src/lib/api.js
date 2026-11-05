import axios from 'axios'

const http = axios.create({ baseURL: '/api', timeout: 30000 })
const unwrap = (request) => request.then(({ data }) => data)

export const api = {
  modelStatus: () => unwrap(http.get('/model/status')),
  essays: () => unwrap(http.get('/essays')),
  essay: (id) => unwrap(http.get(`/essays/${encodeURIComponent(id)}`)),
  upload: (files, theme, motivatingText) => {
    const body = new FormData()
    files.forEach((file) => body.append('files', file))
    body.append('theme', theme)
    body.append('motivating_text', motivatingText)
    return unwrap(http.post('/essays/upload', body, { timeout: 0 }))
  },
  transcribe: (id) => unwrap(http.post(`/essays/${encodeURIComponent(id)}/transcribe`, {}, { timeout: 0 })),
  saveTranscription: (id, body) => unwrap(http.patch(`/essays/${encodeURIComponent(id)}/transcription`, body)),
  correct: (id) => unwrap(http.post(`/essays/${encodeURIComponent(id)}/correct`, {}, { timeout: 0 })),
  correction: (id) => unwrap(http.get(`/essays/${encodeURIComponent(id)}/correction`)),
  deleteEssay: (id) => unwrap(http.delete(`/essays/${encodeURIComponent(id)}`)),
  saveReference: (id, body) => unwrap(http.post(`/essays/${encodeURIComponent(id)}/reference`, body)),
  validationReport: () => unwrap(http.get('/validation/report')),
  rubric: () => unwrap(http.get('/rubric')),
  uploadRubric: (file) => {
    const body = new FormData()
    body.append('file', file)
    return unwrap(http.post('/rubric/upload', body, { timeout: 0 }))
  },
  approveRubric: (body) => unwrap(http.put('/rubric/approve', body, { timeout: 0 })),
}
