import { describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import UploadZone from '../src/components/UploadZone.vue'

describe('UploadZone', () => {
  it('preserves selected order, supports moving pages and removing them', async () => {
    vi.stubGlobal('URL', { createObjectURL: vi.fn(() => 'blob:preview'), revokeObjectURL: vi.fn() })
    const wrapper = mount(UploadZone)
    const input = wrapper.find('input[type="file"]')
    Object.defineProperty(input.element, 'files', { value: [new File(['one'], 'primeira.jpg', { type: 'image/jpeg' }), new File(['two'], 'segunda.png', { type: 'image/png' })] })
    await input.trigger('change')
    expect(wrapper.emitted('change').at(-1)[0].map((file) => file.name)).toEqual(['primeira.jpg', 'segunda.png'])
    await wrapper.find('[aria-label="Mover página 2 para cima"]').trigger('click')
    expect(wrapper.emitted('change').at(-1)[0].map((file) => file.name)).toEqual(['segunda.png', 'primeira.jpg'])
    await wrapper.find('[aria-label="Remover página 1"]').trigger('click')
    expect(wrapper.emitted('change').at(-1)[0].map((file) => file.name)).toEqual(['primeira.jpg'])
    wrapper.unmount()
  })
})
