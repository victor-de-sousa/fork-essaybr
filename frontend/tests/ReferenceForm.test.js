import { describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import ReferenceForm from '../src/components/ReferenceForm.vue'
vi.mock('../src/lib/api.js', () => ({ api: { saveReference: vi.fn().mockResolvedValue({}) } }))
import { api } from '../src/lib/api.js'

describe('teacher reference', () => {
  it('requires all five actual grades and leaves training consent unchecked', async () => {
    const wrapper = mount(ReferenceForm, { props: { essayId: 'essay-1' } })
    expect(wrapper.get('[data-testid="training-consent"]').element.checked).toBe(false)
    expect(wrapper.get('button[type="submit"]').element.disabled).toBe(true)
    for (const select of wrapper.findAll('select')) await select.setValue('160')
    await wrapper.get('form').trigger('submit')
    await flushPromises()
    expect(api.saveReference).toHaveBeenCalledWith('essay-1', { scores: { c1: 160, c2: 160, c3: 160, c4: 160, c5: 160 }, notes: '', consent_training: false })
    expect(wrapper.text()).toContain('Referência humana salva')
  })
})
