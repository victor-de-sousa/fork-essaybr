<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { api } from '../lib/api.js'
import { errorMessage } from '../lib/domain.js'
const route = useRoute()
const router = useRouter()
const essay = ref(null)
const text = ref('')
const theme = ref('')
const motivatingText = ref('')
const confirmed = ref(false)
const loading = ref(true)
const busy = ref('')
const error = ref('')
const message = ref('')
const page = ref(0)
const words = computed(() => text.value.trim() ? text.value.trim().split(/\s+/).length : 0)
const images = computed(() => [...(essay.value?.images || [])].sort((a, b) => a.position - b.position))
watch([text, theme, motivatingText], () => { confirmed.value = false; message.value = '' })
function fill(data) {
  essay.value = data
  text.value = data.transcription || ''
  theme.value = data.theme || ''
  motivatingText.value = data.motivating_text || ''
}
onMounted(async () => {
  try { fill(await api.essay(route.params.id)) }
  catch (err) { error.value = errorMessage(err) }
  finally { loading.value = false }
})
async function transcribe() {
  busy.value = 'ocr'; error.value = ''; message.value = ''
  try { await api.transcribe(route.params.id); fill(await api.essay(route.params.id)); confirmed.value = false; message.value = 'Transcrição concluída. Compare o texto com as páginas antes de confirmar.' }
  catch (err) { error.value = errorMessage(err) }
  finally { busy.value = '' }
}
async function save(evaluate = false) {
  if (busy.value || !text.value.trim() || (evaluate && (!confirmed.value || !theme.value.trim()))) return
  busy.value = evaluate ? 'correct' : 'save'; error.value = ''; message.value = ''
  try {
    essay.value = await api.saveTranscription(route.params.id, { text: text.value, theme: theme.value, motivating_text: motivatingText.value, confirmed: evaluate && confirmed.value })
    if (evaluate) {
      await api.correct(route.params.id)
      await router.push(`/redacoes/${route.params.id}/resultado`)
    } else { confirmed.value = false; message.value = 'Rascunho salvo. Confirme a transcrição quando estiver pronta para avaliar.' }
  } catch (err) { error.value = errorMessage(err) }
  finally { busy.value = '' }
}
</script>

<template>
  <div class="page wide-page">
    <div class="page-heading"><span class="eyebrow">O SEU OLHAR VEM PRIMEIRO</span><h1>Confira cada palavra.</h1><p>O reconhecimento de escrita pode errar. Compare a imagem com o texto e faça os ajustes necessários.</p></div>
    <ol class="workflow"><li class="done"><span>✓</span>Enviar páginas</li><li class="current"><span>2</span>Revisar texto</li><li><span>3</span>Ver avaliação</li></ol>
    <p v-if="loading" class="notice" role="status">Carregando a redação…</p><p v-if="error" class="notice danger" role="alert">{{ error }}</p><p v-if="message" class="notice success" role="status">{{ message }}</p>
    <template v-if="essay">
      <div v-if="essay.ocr_warnings?.length" class="notice warning"><strong>Revise com atenção</strong><ul><li v-for="(warning, index) in essay.ocr_warnings" :key="index">{{ typeof warning === 'string' ? warning : warning.message || JSON.stringify(warning) }}</li></ul></div>
      <div v-if="essay.error && !error" class="notice warning">Última tentativa: {{ essay.error }}</div>
      <div class="review-layout">
        <section class="panel image-panel"><div class="section-heading"><h2>Original</h2><span class="tag">{{ page + 1 }} / {{ images.length }}</span></div><div class="image-stage"><a v-if="images[page]" :href="images[page].url" target="_blank" rel="noopener"><img :src="images[page].url" :alt="`Página ${page + 1} da redação original; abrir em tamanho completo`" /></a><p v-else class="helper">Nenhuma página disponível.</p></div><div class="image-navigation"><button class="button secondary small" :disabled="page === 0" @click="page--">← Anterior</button><small>Clique na imagem para ampliar</small><button class="button secondary small" :disabled="page >= images.length - 1" @click="page++">Próxima →</button></div></section>
        <section class="panel transcription-panel"><div class="section-heading"><div><h2>Transcrição revisável</h2><p class="helper">{{ words }} palavra{{ words === 1 ? '' : 's' }}</p></div><button class="button secondary small" :disabled="!!busy || !!text.trim()" @click="transcribe">{{ busy === 'ocr' ? 'Transcrevendo…' : 'Transcrever páginas' }}</button></div><p v-if="!text" class="helper">Execute o OCR local ou digite a transcrição completa. Nenhum texto será enviado para avaliar sem sua confirmação.</p><label for="transcription" class="visually-hidden">Texto transcrito da redação</label><textarea id="transcription" v-model="text" class="transcription-editor" spellcheck="true" lang="pt-BR" :disabled="!!busy" placeholder="Sua transcrição aparecerá aqui. Você também pode digitá-la manualmente."></textarea><p v-if="text.trim()" class="helper">Para preservar suas revisões, a transcrição automática só está disponível com o campo vazio.</p></section>
      </div>
      <section class="panel review-confirmation"><div class="two-columns"><div class="field"><label for="review-theme">Tema da proposta</label><input id="review-theme" v-model="theme" :disabled="!!busy" /></div><div class="field"><label for="review-motivating">Textos motivadores <span class="optional">opcional</span></label><textarea id="review-motivating" v-model="motivatingText" rows="2" :disabled="!!busy"></textarea></div></div><p v-if="essay.correction" class="notice warning">Salvar alterações inicia uma nova revisão e invalida a avaliação atual. O registro de auditoria é preservado.</p><label class="checkbox-row"><input v-model="confirmed" type="checkbox" :disabled="!!busy || !text.trim()" data-testid="confirmation" /><span><strong>Conferi a transcrição com as imagens.</strong><small>Confirmo que o texto está completo e fiel à minha redação. Alterar o texto exige uma nova confirmação.</small></span></label><div class="form-footer"><button class="button secondary" :disabled="!!busy || !text.trim()" @click="save(false)">{{ busy === 'save' ? 'Salvando…' : 'Salvar rascunho' }}</button><button class="button primary" data-testid="evaluate" :disabled="!!busy || !confirmed || !text.trim() || !theme.trim()" @click="save(true)">{{ busy === 'correct' ? 'Avaliando com os modelos locais…' : 'Confirmar e avaliar' }} <span v-if="busy !== 'correct'" aria-hidden="true">→</span></button></div><p v-if="busy === 'correct' || busy === 'ocr'" class="helper" role="status">O processamento local pode levar alguns minutos. Aguarde nesta página.</p></section>
    </template>
  </div>
</template>
