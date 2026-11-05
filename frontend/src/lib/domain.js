export const competencyNames = {
  c1: 'Domínio da modalidade escrita',
  c2: 'Compreensão do tema',
  c3: 'Construção da argumentação',
  c4: 'Coesão textual',
  c5: 'Proposta de intervenção',
}

export function evidenceSegments(text = '', evidence = []) {
  const ranges = evidence.flatMap((item) => {
    const excerpt = typeof item === 'string' ? item : item?.text || item?.excerpt
    if (!excerpt) return []
    const suppliedStart = Number.isInteger(item.start) ? item.start : -1
    const start = suppliedStart >= 0 && text.slice(suppliedStart, suppliedStart + excerpt.length) === excerpt
      ? suppliedStart : text.indexOf(excerpt)
    return start < 0 ? [] : [[start, start + excerpt.length]]
  }).sort((a, b) => a[0] - b[0])
  const merged = []
  ranges.forEach(([start, end]) => {
    const last = merged.at(-1)
    if (last && start <= last[1]) last[1] = Math.max(end, last[1])
    else merged.push([start, end])
  })
  const parts = []
  let cursor = 0
  merged.forEach(([start, end]) => {
    if (start > cursor) parts.push({ text: text.slice(cursor, start), highlight: false })
    parts.push({ text: text.slice(start, end), highlight: true })
    cursor = end
  })
  if (cursor < text.length || !parts.length) parts.push({ text: text.slice(cursor), highlight: false })
  return parts
}

export function validateImages(files, existingCount = 0) {
  if (files.length + existingCount > 12) return 'Envie no máximo 12 páginas por redação.'
  for (const file of files) {
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) return `${file.name}: use uma imagem PNG, JPEG ou WebP.`
    if (file.size > 20 * 1024 * 1024) return `${file.name}: cada imagem deve ter no máximo 20 MB.`
  }
  return ''
}

export function orderedCompetencies(correction = {}) {
  const entries = Array.isArray(correction.competencies) ? correction.competencies : Object.entries(correction.competencies || {}).map(([id, entry]) => ({ id, ...entry }))
  return Object.entries(competencyNames).map(([id, name]) => {
    const entry = entries.find((item) => item.id?.toLowerCase() === id) || {}
    const score = entry.score ?? correction.scores?.[id]
    return { ...entry, id, name: entry.name || name, score: typeof score === 'number' ? score : null }
  })
}

export function errorMessage(error) {
  const detail = error?.response?.data?.error?.message || error?.response?.data?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) return detail.map((item) => item.msg).join(' ')
  if (error?.code === 'ERR_NETWORK' || !error?.response) return 'Não foi possível conectar ao serviço local. Verifique se o backend está em execução em 127.0.0.1:8000.'
  return error.message || 'Não foi possível concluir esta operação. Tente novamente.'
}

export function formatDate(value) {
  if (!value) return 'Data não informada'
  const date = new Date(value)
  return Number.isNaN(date.valueOf()) ? 'Data não informada' : new Intl.DateTimeFormat('pt-BR', { dateStyle: 'short', timeStyle: 'short' }).format(date)
}

export function statusLabel(status) {
  return ({ uploaded: 'Aguardando transcrição', transcribing: 'Transcrevendo', transcribed: 'Revisar transcrição', confirmed: 'Pronta para avaliar', correcting: 'Em avaliação', corrected: 'Avaliada', completed: 'Avaliada', failed: 'Requer atenção', error: 'Requer atenção' })[status] || 'Em preparação'
}

export function readiness(value) {
  if (typeof value === 'boolean') return value
  return value?.ready === true || value?.available === true || value?.status === 'ready' || value?.status === 'approved'
}
