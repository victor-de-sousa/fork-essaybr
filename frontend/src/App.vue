<script setup>
import { onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { useStatus } from './composables/useStatus.js'
const route = useRoute()
const { refresh } = useStatus()
onMounted(refresh)
</script>

<template>
  <a class="skip-link" href="#main">Ir para o conteúdo</a>
  <div class="app-shell">
    <aside class="sidebar">
      <RouterLink to="/" class="brand" aria-label="Margem, página inicial"><span class="brand-symbol" aria-hidden="true">m<span>·</span></span><div>margem<small>REDAÇÃO ENEM</small></div></RouterLink>
      <div class="workspace-label">SEU ESPAÇO</div>
      <nav aria-label="Navegação principal">
        <RouterLink to="/" exact-active-class="active"><span aria-hidden="true">⌂</span>Visão geral</RouterLink>
        <RouterLink to="/nova" active-class="active"><span aria-hidden="true">＋</span>Nova redação</RouterLink>
        <RouterLink to="/historico" active-class="active"><span aria-hidden="true">▤</span>Minhas redações</RouterLink>
        <RouterLink to="/configuracoes" active-class="active"><span aria-hidden="true">⚙</span>Configurações</RouterLink>
      </nav>
      <div class="sidebar-note"><span class="tiny-star" aria-hidden="true">✦</span><strong>Uma boa escrita começa<br />com uma nova versão.</strong><p>Revisar é parte do processo.</p></div>
      <div class="local-indicator"><span></span>Processamento local<small>Seus textos ficam neste computador.</small></div>
    </aside>
    <div class="workspace">
      <header class="topbar"><span>{{ route.meta.title }}</span><div class="topbar-badge"><span aria-hidden="true">◈</span> Aprender, revisar, evoluir</div></header>
      <main id="main" tabindex="-1"><RouterView :key="route.fullPath" /></main>
      <footer>Uma ferramenta de apoio ao estudo. A avaliação automática não substitui a correção oficial do ENEM.</footer>
    </div>
  </div>
</template>
