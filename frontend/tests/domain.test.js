import { describe, expect, it } from 'vitest'
import { evidenceSegments, validateImages, orderedCompetencies, formatDate, errorMessage } from '../src/lib/domain.js'

describe('evidenceSegments', () => {
  it('preserves original text and marks only exact evidence', () => {
    const text = 'A educação transforma. Educação para todos.'
    const segments = evidenceSegments(text, [{ text: 'educação transforma', start: 2, end: 21 }])
    expect(segments.map((part) => part.text).join('')).toBe(text)
    expect(segments.filter((part) => part.highlight).map((part) => part.text)).toEqual(['educação transforma'])
  })

  it('does not highlight fabricated excerpts or trust incorrect offsets', () => {
    const segments = evidenceSegments('Texto original.', [{ text: 'Texto inventado', start: 0, end: 5 }])
    expect(segments.some((part) => part.highlight)).toBe(false)
  })

  it('merges overlapping evidence without duplicating text', () => {
    const segments = evidenceSegments('abc def ghi', [{ text: 'abc def' }, { text: 'def ghi' }])
    expect(segments).toEqual([{ text: 'abc def ghi', highlight: true }])
  })

  it('keeps HTML as plain text', () => {
    const text = '<img src=x onerror=alert(1)>'
    expect(evidenceSegments(text, [{ text }])).toEqual([{ text, highlight: true }])
  })
})

describe('upload validation', () => {
  it('rejects non-images and oversized images with useful errors', () => {
    expect(validateImages([{ name: 'arquivo.pdf', type: 'application/pdf', size: 20 }])).toMatch(/PNG|JPEG/)
    expect(validateImages([{ name: 'foto.jpg', type: 'image/jpeg', size: 21 * 1024 * 1024 }])).toMatch(/20 MB/)
  })

  it('accepts supported image files', () => {
    expect(validateImages([{ name: 'foto.jpg', type: 'image/jpeg', size: 100 }])).toBe('')
  })
})

describe('report helpers', () => {
  it('never invents unavailable scores', () => {
    const competencies = orderedCompetencies({ scores: { c1: 160 } })
    expect(competencies).toHaveLength(5)
    expect(competencies[0].score).toBe(160)
    expect(competencies[1].score).toBeNull()
  })

  it('presents backend error messages without exposing a generic Axios error', () => {
    expect(errorMessage({ response: { data: { error: { message: 'Modelo ausente.' } } } })).toBe('Modelo ausente.')
  })

  it('handles missing dates', () => {
    expect(formatDate(null)).toBe('Data não informada')
  })
})
