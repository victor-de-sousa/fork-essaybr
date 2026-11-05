<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import { api } from '../lib/api.js'
import { errorMessage, formatDate, orderedCompetencies } from '../lib/domain.js'
import EvidenceText from '../components/EvidenceText.vue'
import ReferenceForm from '../components/ReferenceForm.vue'
const route = useRoute()
const essay = ref(null)
const correction = ref(null)
const loading = ref(true)
const error = ref('')
const selected = ref('c1')
const competencies = computed(() => orderedCompetencies(correction.value || {}))
const active = computed(() => competencies.value.find((entry) => entry.id === selected.value))
const evidence = computed(() => (active.value?.issues || []).map((issue) => issue.evidence).filter(Boolean))
const reference = computed(() => active.value?.citation || active.value?.reference)
onMounted(async () => {
  try { essay.value = await api.essay(route.params.id); correction.value = essay.value.correction || await api.correction(route.params.id) }
  catch (err) { error.value = errorMessage(err) }
  finally { loading.value = false }
})
function download() {
  const blob = new Blob([JSON.stringify({ essay_id: essay.value.id, theme: essay.value.theme, transcription: essay.value.transcription, correction: correction.value }, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url; link.download = `avaliacao-${essay.value.id}.json`; link.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
</script>

<template>
  <div class="page wide-page"><p v-if="loading" class="notice" role="status">Carregando sua avaliação…</p><p v-if="error" class="notice danger" role="alert">{{ error }}</p><template v-if="correction && essay"><div class="page-heading split-heading"><div><span class="eyebrow">DA ANÁLISE À PRÓXIMA VERSÃO</span><h1>Um novo olhar sobre seu texto.</h1><p>{{ essay.theme }}</p><small>{{ formatDate(essay.created_at) }}</small></div><button class="button secondary" @click="download">↓ Exportar relatório</button></div><section class="result-summary"><div class="total-score"><span class="eyebrow light">NOTA ESTIMADA</span><div><strong>{{ correction.total_score ?? '—' }}</strong><span>/ 1000</span></div><p>Avaliação automática de apoio ao estudo</p></div><div class="feedback-summary"><span class="eyebrow">VISÃO GERAL</span><h2>O que sua escrita revela</h2><p>{{ correction.general_feedback || 'Consulte as justificativas por competência abaixo.' }}</p></div></section><div class="competency-grid" role="group" aria-label="Selecionar competência"><button v-for="item in competencies" :key="item.id" class="competency-card" :class="{ selected: selected === item.id }" :aria-pressed="selected === item.id" @click="selected = item.id"><div><span>{{ item.id.toUpperCase() }}</span><strong>{{ item.score ?? '—' }}<small>/200</small></strong></div><h3>{{ item.name }}</h3><div class="score-track"><span :style="{ width: item.score == null ? '0%' : `${item.score / 2}%` }"></span></div></button></div><div class="report-layout"><section class="panel report-text"><div class="section-heading"><div><span class="eyebrow">SEU TEXTO</span><h2>A redação em contexto</h2></div><span class="tag lavender">{{ selected.toUpperCase() }}</span></div><p class="helper">Os destaques mostram evidências citadas na competência selecionada.</p><EvidenceText :text="essay.transcription || ''" :evidence="evidence" /><RouterLink class="text-link" :to="`/redacoes/${essay.id}/revisar`">Revisar a transcrição →</RouterLink></section><section class="panel competency-detail"><span class="eyebrow">{{ active.id.toUpperCase() }} · {{ active.name }}</span><h2>Entenda esta avaliação</h2><p class="justification">{{ active.justification || 'Não há justificativa disponível para esta competência.' }}</p><div v-if="active.issues?.length" class="issues"><h3>Pontos de atenção</h3><article v-for="(issue, index) in active.issues" :key="index" class="issue-card"><strong>{{ issue.kind || `Observação ${index + 1}` }}</strong><blockquote v-if="issue.evidence?.text">“{{ issue.evidence.text }}”</blockquote><p>{{ issue.explanation }}</p><p v-if="issue.suggestion" class="suggestion"><strong>Na próxima versão:</strong> {{ issue.suggestion }}</p></article></div><div v-if="active.strengths?.length" class="strengths"><h3>O que já funciona</h3><ul><li v-for="(strength, index) in active.strengths" :key="index">{{ strength }}</li></ul></div><div v-if="active.improvements?.length" class="improvements"><h3>Para praticar</h3><ul><li v-for="(improvement, index) in active.improvements" :key="index">{{ improvement }}</li></ul></div><div v-if="reference" class="reference-card"><span class="eyebrow">FUNDAMENTO NA CARTILHA</span><p v-if="active.criteria_id"><strong>Critério:</strong> {{ active.criteria_id }}</p><blockquote>“{{ reference.excerpt || reference.text }}”</blockquote><small>Página {{ reference.page }} · {{ correction.rubric_info?.title || 'Cartilha aprovada' }}{{ correction.rubric_info?.year ? ` · ${correction.rubric_info.year}` : '' }}</small></div></section></div><div class="notice warning"><strong>Uma estimativa para orientar seu estudo.</strong><p>{{ correction.disclaimer || 'Esta avaliação automática não representa a nota oficial do ENEM e não substitui a avaliação de um professor. Revise as evidências e os critérios apresentados.' }}</p></div><div class="details-grid"><details class="panel"><summary>Adicionar correção de professor <span>Opcional</span></summary><ReferenceForm :essay-id="essay.id" /></details><details class="panel audit-panel"><summary>Transparência da avaliação <span>Auditoria</span></summary><p class="helper">Registro dos modelos, referência normativa e análises usadas nesta avaliação. Divergências e eventuais revisões ficam disponíveis para consulta.</p><h3>Cartilha utilizada</h3><pre>{{ JSON.stringify(correction.rubric_info || {}, null, 2) }}</pre><h3>Avaliações independentes</h3><pre>{{ JSON.stringify(correction.evaluations || [], null, 2) }}</pre><h3>Divergências e revisão</h3><pre>{{ JSON.stringify(correction.divergences || [], null, 2) }}</pre><template v-if="correction.model_info || correction.prediction"><h3>Modelo supervisionado</h3><pre>{{ JSON.stringify(correction.model_info || correction.prediction, null, 2) }}</pre></template></details></div></template><div v-else-if="!loading" class="empty-state"><h2>A avaliação ainda não está disponível.</h2><p>Confira a transcrição e a configuração dos serviços para continuar.</p><RouterLink class="button primary" :to="`/redacoes/${route.params.id}/revisar`">Voltar à revisão</RouterLink></div></div>
</template>
