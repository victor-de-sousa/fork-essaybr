<script setup>
import { computed, ref } from 'vue'
import { api } from '../lib/api.js'
import { competencyNames, errorMessage } from '../lib/domain.js'
const props = defineProps({ essayId: { type: [String, Number], required: true } })
const scores = ref({ c1: '', c2: '', c3: '', c4: '', c5: '' })
const notes = ref('')
const consent = ref(false)
const busy = ref(false)
const error = ref('')
const saved = ref(false)
const complete = computed(() => Object.values(scores.value).every((value) => value !== '' && [0, 40, 80, 120, 160, 200].includes(Number(value))))
async function submit() {
  if (!complete.value || busy.value) return
  busy.value = true; error.value = ''; saved.value = false
  try { await api.saveReference(props.essayId, { scores: Object.fromEntries(Object.entries(scores.value).map(([key, value]) => [key, Number(value)])), notes: notes.value, consent_training: consent.value }); saved.value = true }
  catch (err) { error.value = errorMessage(err) }
  finally { busy.value = false }
}
</script>

<template>
  <form @submit.prevent="submit"><p class="helper">Se você recebeu uma correção de professor, registre as cinco notas para comparar com a avaliação automática.</p><div class="reference-grades"><div v-for="(name, key) in competencyNames" :key="key" class="field"><label :for="`reference-${key}`" :title="name">{{ key.toUpperCase() }}</label><select :id="`reference-${key}`" v-model="scores[key]" :disabled="busy"><option value="">Nota</option><option v-for="score in [0, 40, 80, 120, 160, 200]" :key="score" :value="score">{{ score }}</option></select></div></div><div class="field"><label for="reference-notes">Observações do professor <span class="optional">opcional</span></label><textarea id="reference-notes" v-model="notes" rows="3" :disabled="busy"></textarea></div><label class="checkbox-row"><input v-model="consent" type="checkbox" data-testid="training-consent" :disabled="busy" /><span><strong>Autorizo o uso desta referência no treinamento local.</strong><small>Opcional. Sem esta autorização, a referência serve apenas para comparação. Textos reservados para teste não podem entrar no treino.</small></span></label><p v-if="error" class="notice danger" role="alert">{{ error }}</p><p v-if="saved" class="notice success" role="status">Referência humana salva. Consulte a comparação na configuração local.</p><button type="submit" class="button secondary" :disabled="busy || !complete">{{ busy ? 'Salvando…' : 'Salvar referência humana' }}</button></form>
</template>
