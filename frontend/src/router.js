import { createRouter, createWebHistory } from 'vue-router'
import HomeView from './views/HomeView.vue'
import NewEssayView from './views/NewEssayView.vue'
import ReviewView from './views/ReviewView.vue'
import ResultView from './views/ResultView.vue'
import HistoryView from './views/HistoryView.vue'
import SettingsView from './views/SettingsView.vue'

export default createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', component: HomeView, meta: { title: 'Seu espaço de escrita' } },
    { path: '/nova', component: NewEssayView, meta: { title: 'Nova redação' } },
    { path: '/redacoes/:id/revisar', component: ReviewView, meta: { title: 'Revisar transcrição' } },
    { path: '/redacoes/:id/resultado', component: ResultView, meta: { title: 'Sua avaliação' } },
    { path: '/historico', component: HistoryView, meta: { title: 'Histórico de redações' } },
    { path: '/configuracoes', component: SettingsView, meta: { title: 'Configuração local' } },
    { path: '/:pathMatch(.*)*', redirect: '/' },
  ],
  scrollBehavior: () => ({ top: 0 }),
})
