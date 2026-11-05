<script setup>
import { computed } from 'vue'
import { evidenceSegments } from '../lib/domain.js'
const props = defineProps({ text: { type: String, default: '' }, evidence: { type: Array, default: () => [] } })
const segments = computed(() => evidenceSegments(props.text, props.evidence))
</script>

<template>
  <div class="essay-text"><template v-for="(segment, index) in segments" :key="index"><mark v-if="segment.highlight">{{ segment.text }}</mark><span v-else>{{ segment.text }}</span></template></div>
</template>
