import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import EvidenceText from '../src/components/EvidenceText.vue'

describe('EvidenceText', () => {
  it('renders untrusted essay content as text, including inside marks', () => {
    const payload = '<img src=x onerror=alert(1)>'
    const wrapper = mount(EvidenceText, { props: { text: `Texto ${payload}`, evidence: [{ text: payload }] } })
    expect(wrapper.find('img').exists()).toBe(false)
    expect(wrapper.find('mark').text()).toBe(payload)
    expect(wrapper.text()).toBe(`Texto ${payload}`)
  })
})
