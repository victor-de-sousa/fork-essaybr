<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { api } from '../lib/api.js'
import { errorMessage } from '../lib/domain.js'
import UploadZone from '../components/UploadZone.vue'
const router = useRouter()
const files = ref([])
const theme = ref('')
const motivatingText = ref('')
const busy = ref(false)
const error = ref('')
async function submit() {
  if (busy.value || !files.value.length || !theme.value.trim()) return
  busy.value = true
  error.value = ''
  try { const essay = await api.upload(files.value, theme.value.trim(), motivatingText.value); await router.push(`/redacoes/${essay.id}/revisar`) }
  catch (err) { error.value = errorMessage(err) }
  finally { busy.value = false }
}
</script>

<template>
  <div class="page">
    <div class="page-heading"><span class="eyebrow">NOVA REDAÇÃO</span><h1>Vamos começar pelo seu texto.</h1><p>Envie as páginas na ordem da leitura. Depois, você poderá conferir a transcrição.</p></div>
    <ol class="workflow"><li class="current"><span>1</span>Enviar páginas</li><li><span>2</span>Revisar texto</li><li><span>3</span>Ver avaliação</li></ol>
    <form class="editor-layout" @submit.prevent="submit">
      <div class="panel"><div class="section-heading"><div><h2>As páginas da sua redação</h2><p class="helper">A imagem deve mostrar o texto completo e estar legível.</p></div><span class="tag">{{ files.length }} página{{ files.length === 1 ? '' : 's' }}</span></div><UploadZone :disabled="busy" @change="files = $event" /><div class="field"><label for="theme">Tema da proposta <span class="required">obrigatório</span></label><input id="theme" v-model="theme" maxlength="2000" required placeholder="Ex.: Desafios para a valorização da cultura brasileira" :disabled="busy" /></div><div class="field"><label for="motivating">Textos motivadores <span class="optional">opcional</span></label><textarea id="motivating" v-model="motivatingText" rows="5" placeholder="Cole os textos motivadores para contextualizar a análise." :disabled="busy"></textarea><small>O texto motivador é contexto da proposta; não deve incluir sua redação.</small></div><p v-if="error" class="notice danger" role="alert">{{ error }}</p><div class="form-footer"><span class="helper">Você confirma o texto antes da avaliação.</span><button type="submit" class="button primary" :disabled="busy || !files.length || !theme.trim()">{{ busy ? 'Enviando páginas…' : 'Continuar para revisão' }} <span v-if="!busy" aria-hidden="true">→</span></button></div></div>
      <aside class="tips-card"><span class="tips-icon" aria-hidden="true">☀</span><h3>Uma boa foto faz diferença.</h3><ul><li>Use luz uniforme e evite sombras.</li><li>Mantenha a câmera paralela à folha.</li><li>Inclua todas as linhas, sem cortes.</li><li>Confira a ordem se houver várias páginas.</li></ul><div class="tips-divider"></div><strong>Seu texto, no seu computador.</strong><p>As imagens e a análise são processadas pelos serviços locais configurados.</p></aside>
    </form>
  </div>
</template>
