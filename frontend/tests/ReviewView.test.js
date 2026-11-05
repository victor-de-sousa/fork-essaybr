import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import ReviewView from '../src/views/ReviewView.vue'

vi.mock('vue-router', () => ({ useRoute: () => ({ params: { id: 'essay-1' } }), useRouter: () => ({ push: vi.fn() }) }))
vi.mock('../src/lib/api.js', () => ({ api: { essay: vi.fn(), saveTranscription: vi.fn(), correct: vi.fn() } }))
import { api } from '../src/lib/api.js'

describe('transcription confirmation', () => {
  beforeEach(() => {
    api.essay.mockResolvedValue({ id: 'essay-1', theme: 'Educação', transcription: 'Minha redação original.', images: [], confirmed: true })
    api.saveTranscription.mockResolvedValue({ id: 'essay-1', theme: 'Educação', transcription: 'Minha redação original.', images: [], confirmed: true })
    api.correct.mockResolvedValue({})
  })

  it('requires fresh explicit confirmation, resets it when text changes', async () => {
    const wrapper = mount(ReviewView)
    await flushPromises()
    expect(wrapper.get('[data-testid="evaluate"]').element.disabled).toBe(true)
    await wrapper.get('[data-testid="confirmation"]').setValue(true)
    expect(wrapper.get('[data-testid="evaluate"]').element.disabled).toBe(false)
    await wrapper.get('#transcription').setValue('Texto revisado com mais cuidado.')
    expect(wrapper.get('[data-testid="confirmation"]').element.checked).toBe(false)
    expect(wrapper.get('[data-testid="evaluate"]').element.disabled).toBe(true)
  })

  it('persists confirmed text before starting grading', async () => {
    const wrapper = mount(ReviewView)
    await flushPromises()
    await wrapper.get('[data-testid="confirmation"]').setValue(true)
    await wrapper.get('[data-testid="evaluate"]').trigger('click')
    await flushPromises()
    expect(api.saveTranscription).toHaveBeenCalledWith('essay-1', expect.objectContaining({ text: 'Minha redação original.', confirmed: true }))
    expect(api.correct).toHaveBeenCalledWith('essay-1')
    expect(api.saveTranscription.mock.invocationCallOrder.at(-1)).toBeLessThan(api.correct.mock.invocationCallOrder.at(-1))
  })

  it('shows validation failure and does not grade after saving fails', async () => {
    api.correct.mockClear()
    api.saveTranscription.mockRejectedValue({ response: { data: { error: { message: 'Transcrição inválida.' } } } })
    const wrapper = mount(ReviewView)
    await flushPromises()
    await wrapper.get('[data-testid="confirmation"]').setValue(true)
    await wrapper.get('[data-testid="evaluate"]').trigger('click')
    await flushPromises()
    expect(wrapper.get('[role="alert"]').text()).toContain('Transcrição inválida.')
    expect(api.correct).not.toHaveBeenCalled()
  })
})
