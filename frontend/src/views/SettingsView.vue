<script setup>
import { computed, onMounted, ref } from 'vue'
import { api } from '../lib/api.js'
import { errorMessage } from '../lib/domain.js'
import { useStatus } from '../composables/useStatus.js'
import StatusPanel from '../components/StatusPanel.vue'
const { refresh } = useStatus()
const rubric = ref(null)
const rubricJson = ref('')
const file = ref(null)
const reviewer = ref('')
const official = ref(false)
const rulesReviewed = ref(false)
const loading = ref(true)
const busy = ref('')
const error = ref('')
const success = ref('')
const report = ref(null)
const reportError = ref('')
const reportLoading = ref(false)
const selectedPage = ref(1)
const pageText = computed(() => rubric.value?.pages?.find((page) => page.page === Number(selectedPage.value))?.text || '')
const template = { year: new Date().getFullYear(), title: '', identity_citation: { page: 1, excerpt: '' }, competencies: ['c1', 'c2', 'c3', 'c4', 'c5'].map((id) => ({ id, name: '', descriptors: [0, 40, 80, 120, 160, 200].map((score) => ({ score, criteria_id: `${id}_${score}`, description: '', citation: { page: 1, excerpt: '' } })) })), special_rules: [], examples: [] }
const jsonError = computed(() => {
  if (!rubricJson.value) return ''
  try { const parsed = JSON.parse(rubricJson.value); if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return 'A rubrica deve ser um objeto JSON.'; return '' }
  catch { return 'O JSON contém um erro de sintaxe. Corrija antes de aprovar.' }
})
const canApprove = computed(() => !!rubric.value?.document_sha256 && official.value && rulesReviewed.value && reviewer.value.trim() && rubricJson.value && !jsonError.value && !busy.value)
function fill(data) {
  rubric.value = data
  rubricJson.value = JSON.stringify(data.rubric || data.proposal || template, null, 2)
  official.value = false; rulesReviewed.value = false
  selectedPage.value = data.pages?.[0]?.page || 1
}
onMounted(async () => {
  try { fill(await api.rubric()) }
  catch (err) { error.value = errorMessage(err) }
  finally { loading.value = false }
})
function selectFile(event) {
  const selected = event.target.files?.[0]
  error.value = ''
  if (selected && !['application/pdf', 'application/json'].includes(selected.type) && !/\.(pdf|json)$/i.test(selected.name)) { error.value = 'Selecione a cartilha em formato PDF ou JSON.'; file.value = null; return }
  file.value = selected || null
}
async function upload() {
  if (!file.value || busy.value) return
  busy.value = 'upload'; error.value = ''; success.value = ''
  try { fill(await api.uploadRubric(file.value)); success.value = 'Fonte normativa recebida. Confira o texto e revise a rubrica antes de aprovar.'; await refresh() }
  catch (err) { error.value = errorMessage(err) }
  finally { busy.value = '' }
}
async function approve() {
  if (!canApprove.value) return
  busy.value = 'approve'; error.value = ''; success.value = ''
  try {
    const data = await api.approveRubric({ document_sha256: rubric.value.document_sha256, official_source_confirmed: official.value, special_rules_reviewed: rulesReviewed.value, reviewer: reviewer.value.trim(), rubric: JSON.parse(rubricJson.value) })
    fill(data); success.value = 'Rubrica aprovada e registrada. Verifique o estado do índice semântico e dos demais serviços.'; await refresh()
  } catch (err) { error.value = errorMessage(err) }
  finally { busy.value = '' }
}
async function loadReport() {
  reportLoading.value = true; reportError.value = ''
  try { report.value = await api.validationReport() }
  catch (err) { reportError.value = errorMessage(err) }
  finally { reportLoading.value = false }
}
</script>

<template>
  <div class="page"><div class="page-heading"><span class="eyebrow">UMA BASE CLARA PARA CADA AVALIAÇÃO</span><h1>Configuração local.</h1><p>Confira os serviços e a referência normativa que orientam a análise da redação.</p></div><StatusPanel /><section class="panel settings-section"><div class="section-heading"><div><span class="eyebrow">REFERÊNCIA NORMATIVA</span><h2>Cartilha do participante</h2></div><span class="tag" :class="{ mint: rubric?.status === 'approved' }">{{ rubric?.status === 'approved' ? 'Aprovada' : rubric?.status === 'pending_review' ? 'Aguardando revisão' : 'Não configurada' }}</span></div><p>Carregue a cartilha oficial do INEP em PDF ou JSON estruturado. O texto extraído e a proposta de rubrica precisam ser conferidos por uma pessoa antes de orientar qualquer nota.</p><div class="rubric-upload"><label class="file-picker" for="rubric-file"><strong>Selecionar PDF ou JSON da cartilha</strong><small>{{ file?.name || 'Escolha o documento no seu computador' }}</small><input id="rubric-file" type="file" accept="application/pdf,application/json,.pdf,.json" :disabled="!!busy" @change="selectFile" /></label><button class="button primary" :disabled="!file || !!busy" @click="upload">{{ busy === 'upload' ? 'Extraindo documento…' : 'Carregar cartilha' }}</button></div><p class="helper">O nome do arquivo não comprova a autenticidade. Confira a origem oficial, o ano, os descritores e todas as regras especiais no próprio documento.</p><p v-if="loading" class="notice" role="status">Carregando referência…</p><p v-if="error" class="notice danger" role="alert">{{ error }}</p><p v-if="success" class="notice success" role="status">{{ success }}</p><template v-if="rubric?.document_sha256"><div class="document-meta"><div><small>Documento</small><strong>{{ rubric.filename }}</strong></div><div><small>Páginas</small><strong>{{ rubric.page_count }}</strong></div><div><small>Versão</small><strong>{{ rubric.version || 'Em revisão' }}</strong></div></div><details class="hash-details"><summary>Identificador do documento</summary><code>{{ rubric.document_sha256 }}</code></details><p v-if="rubric.extraction_error" class="notice warning">{{ rubric.extraction_error }} Você pode preencher e revisar a rubrica manualmente com base nas páginas extraídas.</p><p v-if="rubric.embedding_status?.error" class="notice warning">Índice semântico: {{ rubric.embedding_status.error }}</p><div class="rubric-editor-grid"><section><div class="section-heading"><h3>Texto extraído do PDF</h3><label class="page-selector">Página <select v-model="selectedPage"><option v-for="page in rubric.pages || []" :key="page.page" :value="page.page">{{ page.page }}</option></select></label></div><div class="pdf-text">{{ pageText || 'Não há texto extraído nesta página. Verifique se o PDF contém texto legível.' }}</div></section><section><div class="section-heading"><h3>Rubrica estruturada</h3><span class="tag">JSON editável</span></div><label class="visually-hidden" for="rubric-json">Rubrica em JSON</label><textarea id="rubric-json" v-model="rubricJson" class="json-editor" spellcheck="false" :disabled="!!busy" @input="official = false; rulesReviewed = false"></textarea><p v-if="jsonError" class="notice danger" role="alert">{{ jsonError }}</p></section></div><details class="schema-help"><summary>O que revisar na rubrica</summary><p>Confirme o ano e o título em <code>identity_citation</code>. Cada competência precisa de seis descritores, com notas de 0 a 200 em passos de 40, critérios e citações literais com página. Revise <code>special_rules</code> para contemplar anulação, limites e exceções presentes no documento, incluindo qualquer condição que zere a redação. As citações são verificadas pelo backend.</p><p>Uma proposta automática ou um modelo vazio não está aprovado. Preencha todos os campos a partir da cartilha oficial; a API bloqueia rubricas incompletas ou referências inexistentes.</p></details><div class="field"><label for="reviewer">Responsável pela revisão da cartilha</label><input id="reviewer" v-model="reviewer" placeholder="Nome de quem conferiu o documento" :disabled="!!busy" /></div><label class="checkbox-row"><input v-model="official" type="checkbox" :disabled="!!busy" /><span><strong>Conferi que este é o documento oficial e que o ano corresponde ao conteúdo.</strong><small>A confirmação refere-se ao PDF identificado acima e às citações da rubrica revisada.</small></span></label><label class="checkbox-row"><input v-model="rulesReviewed" type="checkbox" :disabled="!!busy" /><span><strong>Revisei todos os descritores, regras especiais e condições de anulação.</strong><small>Conferi a completude da rubrica e a fidelidade das citações à cartilha.</small></span></label><div class="form-footer"><span class="helper">A aprovação será registrada com o responsável e a versão.</span><button class="button primary" :disabled="!canApprove" @click="approve">{{ busy === 'approve' ? 'Validando e aprovando…' : 'Aprovar rubrica revisada' }}</button></div></template></section><section class="panel settings-section"><div class="section-heading"><div><span class="eyebrow">AVALIAÇÃO DO SISTEMA</span><h2>Comparação com correções humanas</h2></div><button class="button secondary small" :disabled="reportLoading" @click="loadReport">{{ reportLoading ? 'Carregando…' : 'Consultar relatório' }}</button></div><p class="helper">As métricas são calculadas a partir das referências reais cadastradas. A autorização para treinamento é separada da comparação.</p><p v-if="reportError" class="notice danger" role="alert">{{ reportError }}</p><pre v-if="report" class="validation-report">{{ JSON.stringify(report, null, 2) }}</pre><p v-else class="helper">Adicione a correção de um professor na página de uma avaliação para enriquecer a comparação.</p></section><section class="setup-note"><h3>Modelos ausentes?</h3><p>Os serviços usam arquivos e modelos locais. O backend precisa de um modelo supervisionado treinado, do OCR TrOCR para transcrição automática, de embeddings para consultar a cartilha e do Ollama em execução. A instalação, o download explícito dos modelos e os comandos de treinamento estão no guia de operação do projeto. A interface não baixa modelos automaticamente.</p></section></div>
</template>
