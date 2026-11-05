<script setup>
import { computed, onMounted, ref } from 'vue'
import { api } from '../lib/api.js'
import { errorMessage, formatDate, statusLabel } from '../lib/domain.js'
import StatusPanel from '../components/StatusPanel.vue'
const essays = ref([])
const loading = ref(true)
const error = ref('')
const completed = computed(() => essays.value.filter((essay) => essay.correction || ['corrected', 'completed'].includes(essay.status)))
onMounted(async () => {
  try { essays.value = await api.essays() }
  catch (err) { error.value = errorMessage(err) }
  finally { loading.value = false }
})
</script>

<template>
  <div class="page home-page">
    <section class="hero">
      <div class="hero-content"><span class="eyebrow light">CADA VERSÃO, UM PASSO ADIANTE</span><h1>Seu texto tem muito<br />a dizer<span>.</span></h1><p>Entenda sua escrita pelas cinco competências do ENEM. Da folha ao próximo rascunho, com você no controle.</p><RouterLink class="button white" to="/nova">Avaliar uma redação <span aria-hidden="true">↗</span></RouterLink><div class="hero-footnote">Transcrição revisável · Evidências no texto · Cartilha de referência</div></div>
      <div class="hero-art" aria-hidden="true"><span class="art-spark spark-one">✳</span><div class="paper-sheet"><div class="paper-top"><span>MINHA PRÓXIMA VERSÃO</span><span>↗</span></div><div class="paper-line"></div><div class="paper-line short"></div><div class="paper-line highlighted"></div><div class="paper-line"></div><div class="paper-line medium"></div><div class="paper-line"></div><div class="paper-line short"></div><div class="paper-note">Ideias que ganham forma.</div></div><span class="art-spark spark-two">✦</span><div class="revision-pill"><span>✓</span> Revisar faz parte.</div></div>
    </section>

    <div class="summary-grid"><article class="summary-card"><span class="summary-icon lavender" aria-hidden="true">▤</span><div><strong>{{ loading || error ? '—' : essays.length }}</strong><span>redações no seu espaço</span></div></article><article class="summary-card"><span class="summary-icon mint" aria-hidden="true">✓</span><div><strong>{{ loading || error ? '—' : completed.length }}</strong><span>avaliações concluídas</span></div></article><article class="summary-card"><span class="summary-icon peach" aria-hidden="true">✦</span><div><strong>5 competências</strong><span>para orientar sua evolução</span></div></article></div>

    <div class="section-heading"><div><span class="eyebrow">DO PAPEL À PRÓXIMA VERSÃO</span><h2>Um processo claro, em três passos</h2></div><span class="tag">No seu ritmo</span></div>
    <div class="steps-grid"><article class="step-card"><span class="step-number">01</span><h3>Envie sua redação</h3><p>Fotografe ou digitalize as páginas e informe o tema da proposta.</p></article><article class="step-card"><span class="step-number">02</span><h3>Revise a transcrição</h3><p>Compare o texto com a imagem e confirme cada palavra antes da análise.</p></article><article class="step-card"><span class="step-number">03</span><h3>Entenda o que melhorar</h3><p>Veja notas, trechos comentados e orientações fundamentadas na cartilha.</p></article></div>

    <section class="recent-section"><div class="section-heading"><div><span class="eyebrow">CONTINUE DE ONDE PAROU</span><h2>Redações recentes</h2></div><RouterLink to="/historico" class="text-link">Ver todas →</RouterLink></div><p v-if="error" class="notice danger" role="alert">{{ error }}</p><p v-else-if="loading" class="notice" role="status">Carregando suas redações…</p><div v-else-if="!essays.length" class="empty-state compact"><span aria-hidden="true">✎</span><div><h3>Uma página em branco é um começo.</h3><p>Sua primeira redação aparecerá aqui. Que tal começar agora?</p></div><RouterLink class="button secondary" to="/nova">Enviar redação</RouterLink></div><div v-else class="essay-list"><RouterLink v-for="essay in essays.slice(0, 3)" :key="essay.id" :to="`/redacoes/${essay.id}/${essay.correction || ['corrected', 'completed'].includes(essay.status) ? 'resultado' : 'revisar'}`" class="essay-row"><span class="document-icon" aria-hidden="true">▤</span><div class="essay-row-main"><strong>{{ essay.theme || 'Redação sem tema' }}</strong><small>{{ formatDate(essay.created_at) }}</small></div><span class="tag">{{ statusLabel(essay.status) }}</span><span aria-hidden="true">↗</span></RouterLink></div></section>
    <StatusPanel compact />
  </div>
</template>
