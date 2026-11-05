<script setup>
import { computed } from 'vue'
import { useStatus } from '../composables/useStatus.js'
import { readiness } from '../lib/domain.js'
defineProps({ compact: Boolean })
const { status, loading, error, refresh } = useStatus()
const labels = { scoring: 'Modelo de notas', ocr: 'Transcrição OCR', ollama: 'Ollama local', rubric: 'Cartilha aprovada' }
const entries = computed(() => Object.entries(labels).map(([key, label]) => ({ key, label, ready: readiness(status.value?.[key]), detail: status.value?.[key]?.error || status.value?.[key]?.message || status.value?.[key]?.reason || '' })))
</script>

<template>
  <section class="status-panel" :class="{ compact }" aria-label="Estado dos serviços locais">
    <div class="section-heading"><div><span class="eyebrow">Ambiente local</span><h2>Pronto para escrever?</h2></div><button class="button ghost small" :disabled="loading" @click="refresh">{{ loading ? 'Verificando…' : 'Atualizar' }}</button></div>
    <p v-if="error" class="notice danger" role="alert">{{ error }}</p>
    <p v-else-if="loading && !status" class="helper" role="status">Verificando os serviços…</p>
    <div v-else class="service-grid">
      <div v-for="entry in entries" :key="entry.key" class="service-item">
        <span class="status-dot" :class="{ ready: entry.ready }" aria-hidden="true"></span>
        <div><strong>{{ entry.label }}</strong><small>{{ entry.ready ? 'Disponível' : 'Configuração necessária' }}</small><p v-if="entry.detail && !compact" class="helper">{{ entry.detail }}</p></div>
      </div>
    </div>
    <p v-if="status && entries.some((entry) => !entry.ready)" class="helper">Para avaliar, prepare o modelo supervisionado, o Ollama e a cartilha. O OCR permite transcrever as imagens automaticamente. <RouterLink to="/configuracoes">Ver configuração →</RouterLink></p>
  </section>
</template>
