<script setup>
import { computed, onMounted, ref } from 'vue'
import { api } from '../lib/api.js'
import { errorMessage, formatDate, statusLabel } from '../lib/domain.js'
const essays = ref([])
const loading = ref(true)
const error = ref('')
const query = ref('')
const deleting = ref(null)
const busy = ref(false)
const filtered = computed(() => essays.value.filter((essay) => (essay.theme || '').toLocaleLowerCase('pt-BR').includes(query.value.toLocaleLowerCase('pt-BR'))))
async function load() {
  loading.value = true; error.value = ''
  try { essays.value = await api.essays() }
  catch (err) { error.value = errorMessage(err) }
  finally { loading.value = false }
}
async function remove() {
  busy.value = true; error.value = ''
  try { await api.deleteEssay(deleting.value.id); essays.value = essays.value.filter((essay) => essay.id !== deleting.value.id); deleting.value = null }
  catch (err) { error.value = errorMessage(err) }
  finally { busy.value = false }
}
onMounted(load)
</script>

<template>
  <div class="page"><div class="page-heading split-heading"><div><span class="eyebrow">O CAMINHO DA SUA ESCRITA</span><h1>Minhas redações.</h1><p>Revisite suas ideias, retome uma revisão e acompanhe suas avaliações.</p></div><RouterLink to="/nova" class="button primary">＋ Nova redação</RouterLink></div><div class="search-row"><label for="history-search" class="visually-hidden">Buscar pelo tema</label><input id="history-search" v-model="query" type="search" placeholder="Buscar pelo tema da redação…" /><button class="button secondary" :disabled="loading" @click="load">Atualizar</button></div><p v-if="error" class="notice danger" role="alert">{{ error }}</p><p v-if="loading" class="notice" role="status">Carregando histórico…</p><section v-else-if="!filtered.length" class="empty-state"><span aria-hidden="true">▤</span><h2>{{ query ? 'Nenhuma redação encontrada.' : 'Sua história começa na primeira página.' }}</h2><p>{{ query ? 'Tente outro tema na busca.' : 'Envie uma redação para iniciar seu histórico de revisão.' }}</p><RouterLink v-if="!query" class="button primary" to="/nova">Enviar minha primeira redação</RouterLink></section><div v-else class="history-grid"><article v-for="essay in filtered" :key="essay.id" class="panel history-card"><div class="section-heading"><span class="document-icon" aria-hidden="true">▤</span><span class="tag" :class="{ mint: essay.correction }">{{ statusLabel(essay.status) }}</span></div><h2>{{ essay.theme || 'Redação sem tema' }}</h2><p class="helper">{{ formatDate(essay.created_at) }}</p><div v-if="essay.correction?.total_score != null" class="history-score"><strong>{{ essay.correction.total_score }}</strong><span>/ 1000 pontos</span></div><div class="history-actions"><RouterLink class="button secondary small" :to="`/redacoes/${essay.id}/${essay.correction || ['corrected', 'completed'].includes(essay.status) ? 'resultado' : 'revisar'}`">{{ essay.correction || ['corrected', 'completed'].includes(essay.status) ? 'Ver avaliação' : 'Continuar revisão' }} →</RouterLink><button class="button ghost small danger-text" :aria-label="`Excluir redação: ${essay.theme || essay.id}`" @click="deleting = essay">Excluir</button></div></article></div><div v-if="deleting" class="modal-backdrop" @keydown.esc="!busy && (deleting = null)"><section class="modal panel" role="alertdialog" aria-modal="true" aria-labelledby="delete-title" aria-describedby="delete-description"><h2 id="delete-title">Excluir esta redação?</h2><p id="delete-description">“{{ deleting.theme || 'Redação sem tema' }}” e suas imagens, avaliação e referências serão removidas do armazenamento local. Esta ação não pode ser desfeita.</p><div class="form-footer"><button class="button secondary" :disabled="busy" @click="deleting = null">Cancelar</button><button class="button danger-button" :disabled="busy" @click="remove">{{ busy ? 'Excluindo…' : 'Excluir redação' }}</button></div></section></div></div>
</template>
