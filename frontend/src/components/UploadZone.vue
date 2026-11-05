<script setup>
import { ref, onBeforeUnmount } from 'vue'
import { validateImages } from '../lib/domain.js'

const props = defineProps({ disabled: Boolean })
const emit = defineEmits(['change'])
const input = ref(null)
const pages = ref([])
const error = ref('')
const dragging = ref(false)
let nextId = 0

function addFiles(files) {
  if (props.disabled) return
  const incoming = Array.from(files || [])
  error.value = validateImages(incoming, pages.value.length)
  if (error.value) return
  pages.value.push(...incoming.map((file) => ({ file, url: URL.createObjectURL(file), id: nextId++ })))
  emit('change', pages.value.map((page) => page.file))
}
function changed(event) { addFiles(event.target.files); event.target.value = '' }
function drop(event) { dragging.value = false; addFiles(event.dataTransfer.files) }
function move(index, offset) {
  const target = index + offset
  if (target < 0 || target >= pages.value.length) return
  const [page] = pages.value.splice(index, 1)
  pages.value.splice(target, 0, page)
  emit('change', pages.value.map((item) => item.file))
}
function remove(index) {
  const [page] = pages.value.splice(index, 1)
  URL.revokeObjectURL(page.url)
  emit('change', pages.value.map((item) => item.file))
}
onBeforeUnmount(() => pages.value.forEach((page) => URL.revokeObjectURL(page.url)))
</script>

<template>
  <div>
    <button type="button" class="upload-zone" :class="{ dragging }" :disabled="disabled" @click="input.click()" @dragover.prevent="dragging = true" @dragleave.prevent="dragging = false" @drop.prevent="drop">
      <span class="upload-symbol" aria-hidden="true">↥</span>
      <strong>Traga sua redação para cá</strong>
      <span>Arraste as imagens ou <span class="text-accent">escolha os arquivos</span></span>
      <small>PNG, JPEG ou WebP · até 20 MB por página · máximo de 12 páginas</small>
    </button>
    <input ref="input" class="visually-hidden" type="file" multiple accept="image/jpeg,image/png,image/webp" aria-label="Selecionar imagens da redação" :disabled="disabled" @change="changed" />
    <p v-if="error" class="notice danger" role="alert">{{ error }}</p>
    <div v-if="pages.length" class="page-grid">
      <article v-for="(page, index) in pages" :key="page.id" class="page-preview">
        <img :src="page.url" :alt="`Prévia da página ${index + 1}`" />
        <div class="page-caption"><strong>Página {{ index + 1 }}</strong><small :title="page.file.name">{{ page.file.name }}</small></div>
        <div class="page-controls">
          <button type="button" class="icon-button" :aria-label="`Mover página ${index + 1} para cima`" :disabled="disabled || index === 0" @click="move(index, -1)">↑</button>
          <button type="button" class="icon-button" :aria-label="`Mover página ${index + 1} para baixo`" :disabled="disabled || index === pages.length - 1" @click="move(index, 1)">↓</button>
          <button type="button" class="icon-button remove" :aria-label="`Remover página ${index + 1}`" :disabled="disabled" @click="remove(index)">×</button>
        </div>
      </article>
    </div>
    <p v-if="pages.length" class="helper">Confira a ordem das páginas antes de continuar. As setas alteram a sequência da transcrição.</p>
  </div>
</template>
