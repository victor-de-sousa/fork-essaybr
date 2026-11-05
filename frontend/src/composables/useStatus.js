import { ref } from 'vue'
import { api } from '../lib/api.js'
import { errorMessage } from '../lib/domain.js'

const status = ref(null)
const loading = ref(false)
const error = ref('')

export function useStatus() {
  async function refresh() {
    if (loading.value) return
    loading.value = true
    error.value = ''
    try { status.value = await api.modelStatus() }
    catch (err) { error.value = errorMessage(err); status.value = null }
    finally { loading.value = false }
  }
  return { status, loading, error, refresh }
}
