<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { api, streamChat } from './api'

type Message = { role: 'user' | 'assistant'; content: string; status?: string; citations?: any[] }
type Attraction = Record<string, any> & { attraction_id: string; name: string }

const navItems = [
  { id: 'chat', label: '智能咨询' },
  { id: 'attractions', label: '景点浏览' },
  { id: 'map', label: '景区地图' },
  { id: 'notices', label: '最新公告' },
  { id: 'recommend', label: '路线推荐' },
  { id: 'admin', label: '运营后台' },
] as const

const active = ref<(typeof navItems)[number]['id']>('chat')
const messages = ref<Message[]>([
  { role: 'assistant', content: '你好，我是九寨沟景区智能助手。可以咨询景点、门票预约、观光车换乘和游览路线。' },
])
const input = ref('')
const loading = ref(false)
// Keep the active conversation only in memory. A fresh page load starts with
// the welcome message instead of restoring a previous browser conversation.
const conversationId = ref<string>()
const citations = ref<any[]>([])
const attractions = ref<Attraction[]>([])
const notices = ref<any[]>([])
const recommendation = ref<any>()
const feedback = ref('')
const feedbackStatus = ref('')
const feedbackError = ref('')
// Admin authentication is an HttpOnly cookie. Keep only a non-secret UI flag.
const adminToken = ref('')
// Keep the identifier between unlocks; never keep the password.
const adminEmail = ref(localStorage.getItem('admin_email') || '')
const adminPassword = ref('')
const dashboard = ref<any>()
const candidates = ref<any[]>([])
const adminLoading = ref(false)
const adminError = ref('')
const reviewBusy = ref<string | null>(null)
const reviewNotice = ref('')
const reviewError = ref('')
const weather = ref<any>()
const weatherLoading = ref(true)
const attractionsLoading = ref(true)
const attractionsError = ref('')
const noticesLoading = ref(false)
const recDuration = ref(240)
const recGroup = ref('普通游客')
const recPreference = ref('湖泊/海子')
const recommendationLoading = ref(false)
const recommendationError = ref('')
const selectedAttraction = ref<Attraction | null>(null)
const detailLoading = ref(false)
const detailError = ref('')
const imageErrors = ref<Record<string, boolean>>({})
const mapImage = '/jiuzhaigou-map.png'
const mapScale = ref(1)

const attractionImages: Record<string, string> = {
  attr_002: 'https://images.unsplash.com/photo-1500534623283-312aade485b7?auto=format&fit=crop&w=1200&q=82',
  attr_004: '/attractions/attr_004-nuorilang-waterfall.jpg',
  attr_007: '/attractions/attr_007-shuzheng-lakes.jpg',
  attr_011: '/attractions/attr_011-reed-lake.jpg',
  attr_014: '/attractions/attr_014-pearl-shoal-waterfall.jpg',
  attr_016: '/attractions/attr_016-mirror-lake.jpg',
  attr_021: '/attractions/attr_021-five-flower-lake.jpg',
  attr_022: '/attractions/attr_022-panda-lake.jpg',
  attr_023: '/attractions/attr_023-panda-waterfall.jpg',
  attr_024: '/attractions/attr_024-arrow-bamboo-lake.jpg',
  attr_031: '/attractions/attr_031-long-lake.jpg',
  attr_032: '/attractions/attr_032-five-color-pond.jpg',
}
const categoryImages: Record<string, string> = {
  '瀑布': 'https://images.unsplash.com/photo-1433086966358-54859d0ed716?auto=format&fit=crop&w=1200&q=82',
  '湖泊/海子': 'https://images.unsplash.com/photo-1439853949127-fa647821eba0?auto=format&fit=crop&w=1200&q=82',
  '森林': 'https://images.unsplash.com/photo-1448375240586-882707db888b?auto=format&fit=crop&w=1200&q=82',
  '藏寨人文': 'https://images.unsplash.com/photo-1518005020951-eccb494ad742?auto=format&fit=crop&w=1200&q=82',
}

const weatherIcon = computed(() => {
  if (!weather.value?.available) return '—'
  const label = String(weather.value.weather || '')
  if (label.includes('雨')) return '雨'
  if (label.includes('雪')) return '雪'
  if (label.includes('云') || label.includes('阴')) return '云'
  return '晴'
})
const routeStops = computed<Attraction[]>(() => recommendation.value?.attractions || [])
const routeSegments = computed<any[]>(() => {
  const existing = new Map(
    (recommendation.value?.route_segments || []).map((segment: any) => [
      `${segment.from}|${segment.to}`,
      segment,
    ]),
  )
  return routeStops.value.slice(1).map((stop, index) => ({
    from: routeStops.value[index]?.name,
    to: stop.name,
    ...(existing.get(`${routeStops.value[index]?.name}|${stop.name}`) || {}),
    route_type:
      existing.get(`${routeStops.value[index]?.name}|${stop.name}`)?.route_type ||
      (routeStops.value[index]?.valley === stop.valley ? '步行栈道' : '观光车换乘'),
    estimated_minutes: existing.get(`${routeStops.value[index]?.name}|${stop.name}`)?.estimated_minutes ?? null,
  }))
})

function sourceLabel(c: any) {
  return c?.authority === 'realtime'
    ? '实时数据'
    : c?.source_type === 'facility'
      ? '游客服务'
      : ['attraction', 'faq', 'park', 'route'].includes(c?.source_type)
        ? '景区知识'
        : c?.source_type === 'notice' ? '官方公告' : '景区资料'
}

function imageFor(item: Attraction) {
  return attractionImages[item.attraction_id] || categoryImages[item.category] || categoryImages['森林']
}

function markImageError(id: string) {
  imageErrors.value = { ...imageErrors.value, [id]: true }
}

function zoomMap(delta: number) {
  mapScale.value = Math.min(2.6, Math.max(1, Number((mapScale.value + delta).toFixed(2))))
}

function resetMapZoom() {
  mapScale.value = 1
}

async function send() {
  if (!input.value.trim() || loading.value) return
  const text = input.value.trim()
  input.value = ''
  messages.value.push({ role: 'user', content: text }, { role: 'assistant', content: '正在准备回答…' })
  loading.value = true
  try {
    await streamChat(text, (event, data) => {
      if (event === 'status') messages.value.at(-1)!.content = String(data || '正在处理')
      if (event === 'citations') {
        citations.value = data
        messages.value.at(-1)!.citations = data
      }
      if (event === 'token') {
        if (!messages.value.at(-1)!.content || messages.value.at(-1)!.content.includes('正在')) messages.value.at(-1)!.content = ''
        messages.value.at(-1)!.content += data
      }
      if (event === 'result') {
        messages.value.at(-1)!.status = ''
        conversationId.value = data.conversation_id
      }
    }, conversationId.value)
  } catch (error: any) {
    messages.value.at(-1)!.content = `请求失败：${error.message || '服务暂不可用'}`
  } finally {
    loading.value = false
  }
}

async function loadAttractions() {
  attractionsLoading.value = true
  attractionsError.value = ''
  try {
    const data = await api('/attractions?limit=60')
    attractions.value = data.items || []
  } catch (error: any) {
    attractionsError.value = error.message || '景点数据暂时无法读取'
  } finally {
    attractionsLoading.value = false
  }
}

async function openAttraction(item: Attraction) {
  selectedAttraction.value = item
  detailError.value = ''
  detailLoading.value = true
  try {
    const detail = await api(`/attractions/${encodeURIComponent(item.attraction_id)}`)
    selectedAttraction.value = { ...item, ...detail }
  } catch (error: any) {
    detailError.value = error.message || '详情暂时无法读取，将显示景点目录信息。'
  } finally {
    detailLoading.value = false
  }
}

function closeAttraction() {
  selectedAttraction.value = null
  detailError.value = ''
}

async function loadNotices() {
  noticesLoading.value = true
  try {
    const data = await api('/notices?active_only=true&limit=20')
    notices.value = data.items || []
  } finally {
    noticesLoading.value = false
  }
}

async function loadWeather() {
  weatherLoading.value = true
  try {
    weather.value = await api('/weather')
  } catch {
    weather.value = { available: false, message: '天气服务暂不可用' }
  } finally {
    weatherLoading.value = false
  }
}

async function recommend() {
  recommendationLoading.value = true
  recommendationError.value = ''
  try {
    recommendation.value = await api('/recommendations', {
      method: 'POST',
      body: JSON.stringify({
        duration_minutes: recDuration.value,
        groups: [recGroup.value],
        preferences: [recPreference.value],
        weather: weather.value?.available ? weather.value.weather : '晴',
      }),
    })
  } catch (error: any) {
    recommendationError.value = error.message || '路线生成失败，请稍后重试。'
  } finally {
    recommendationLoading.value = false
  }
}

async function submitFeedback() {
  if (!feedback.value.trim()) return
  feedbackStatus.value = ''
  feedbackError.value = ''
  try {
    await api('/feedbacks', { method: 'POST', body: JSON.stringify({ content: feedback.value, feedback_type: '建议' }) })
    feedback.value = ''
    feedbackStatus.value = '反馈已提交，感谢你的建议。'
  } catch (error: any) {
    feedbackError.value = error.message || '反馈提交失败，请稍后重试。'
  }
}

async function login() {
  adminError.value = ''
  adminLoading.value = true
  try {
    const data = await api('/admin/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email: adminEmail.value, password: adminPassword.value }),
    })
    localStorage.setItem('admin_email', adminEmail.value.trim())
    adminToken.value = 'cookie'
    adminPassword.value = ''
    await loadAdmin()
  } catch (error: any) {
    adminError.value = error.message || '登录失败，请检查账号和密码。'
  } finally {
    adminLoading.value = false
  }
}

// Lock the admin area whenever the SPA leaves it. Clearing the UI state alone
// would only hide the page, so also clear the browser session cookie through
// the backend. Returning to the admin section then requires a fresh password.
function selectSection(next: (typeof navItems)[number]['id']) {
  if (active.value === 'admin' && next !== 'admin') {
    const hadAdminSession = adminToken.value === 'cookie'
    adminToken.value = ''
    adminPassword.value = ''
    dashboard.value = undefined
    candidates.value = []
    adminError.value = ''
    reviewNotice.value = ''
    reviewError.value = ''
    if (hadAdminSession) {
      void api('/admin/auth/logout', { method: 'POST' }).catch(() => undefined)
    }
  }
  active.value = next
}

async function loadAdmin() {
  adminLoading.value = true
  adminError.value = ''
  try {
    dashboard.value = await api('/admin/dashboard')
    candidates.value = (await api('/admin/feedback-candidates')).items || []
  } catch (error: any) {
    adminError.value = error.message || '后台数据暂时无法读取，请重新登录。'
    if (String(error.message || '').includes('401')) {
      adminToken.value = ''
    }
  } finally {
    adminLoading.value = false
  }
}

async function review(id: string, action: 'approve' | 'reject') {
  if (reviewBusy.value) return
  reviewBusy.value = id
  reviewNotice.value = ''
  reviewError.value = ''
  try {
    const result = await api(`/admin/feedback-candidates/${encodeURIComponent(id)}/${action}`, {
      method: 'POST',
    })
    reviewNotice.value = action === 'approve'
      ? `已采纳反馈，知识文档已创建，向量同步任务 ${result.task_id || ''} 已入队；同步完成后即可检索。`
      : '已驳回该反馈。'
    await loadAdmin()
  } catch (error: any) {
    reviewError.value = error.message || '审核操作失败，请刷新后重试。'
  } finally {
    reviewBusy.value = null
  }
}

onMounted(() => {
  loadAttractions()
  loadNotices()
  loadWeather()
  // Do not probe or restore an admin session in the background. Every entry
  // into the admin section is an explicit password unlock.
})
</script>

<template>
  <div class="shell">
    <header class="topbar">
      <div class="brand"><span class="brand-mark">JZ</span><div><strong>九寨沟景区</strong><small>智能服务与运营中心</small></div></div>
      <nav aria-label="主导航"><button v-for="item in navItems" :key="item.id" :class="{ active: active === item.id }" @click="selectSection(item.id)">{{ item.label }}</button></nav>
      <span class="status-dot">● 在线</span>
    </header>

    <main v-if="active === 'chat'" class="content chat-layout">
      <aside class="side-panel"><p class="eyebrow">JIUZHAIGOU / 01</p><h1>把九寨的<br><em>色彩交给内行</em></h1><p class="muted">官方知识、门票预约提醒与观光车换乘建议，集中在一个可靠的景区助手里。</p></aside>
      <section class="chat-panel"><div class="section-head"><div><p class="eyebrow">OFFICIAL ASSISTANT</p><h2>景区智能咨询</h2></div><span class="live-pill">实时连接</span></div><div class="messages"><article v-for="(message, index) in messages" :key="index" :class="['message', message.role]"><div class="avatar">{{ message.role === 'assistant' ? 'JZ' : '你' }}</div><div><div class="bubble">{{ message.content }}</div><div v-if="message.citations?.length" class="citations"><span v-for="citation in message.citations.slice(0, 3)" :key="citation.document_id || citation.source_id">{{ sourceLabel(citation) }} · {{ citation.updated_at ? `更新于 ${citation.updated_at.slice(0, 10)}` : '当前知识' }}</span></div></div></article></div><form class="composer" @submit.prevent="send"><input v-model="input" placeholder="问问九寨沟景区的任何事…" :disabled="loading"><button :disabled="loading">{{ loading ? '发送中…' : '发送 ↗' }}</button></form></section>
      <aside class="info-panel"><div class="info-card dark"><p class="eyebrow">TODAY AT JIUZHAIGOU</p><strong>九寨沟风景名胜区</strong><p>旺季 08:00 — 18:00</p><div class="weather" aria-live="polite"><span class="weather-icon">{{ weatherLoading ? '…' : weatherIcon }}</span><div v-if="weatherLoading" class="weather-copy"><b>正在读取天气</b><small>高德天气服务</small></div><div v-else-if="weather?.available" class="weather-copy"><b>{{ weather.temperature }}°C · {{ weather.weather }}</b><small>{{ weather.wind_direction }}风 {{ weather.wind_power }}级 · 湿度 {{ weather.humidity }}%</small><small>更新于 {{ weather.report_time || '高德实时数据' }}</small></div><div v-else class="weather-copy"><b>天气暂不可用</b><small>{{ weather?.message || '请以现场公告为准' }}</small></div></div></div><div class="info-card"><p class="eyebrow">ANSWER SOURCES</p><div v-if="citations.length" class="source-list"><div v-for="citation in citations.slice(0, 4)" :key="citation.source_id"><b>{{ sourceLabel(citation) }}</b><small>{{ citation.updated_at ? `更新于 ${citation.updated_at.slice(0, 10)}` : '更新时间未提供' }} · {{ citation.authority === 'realtime' ? '实时' : '非实时资料' }}</small></div></div><p v-else class="muted">回答后显示来源类型与更新时间</p></div><div class="feedback-box"><p class="eyebrow">帮助我们变好</p><textarea v-model="feedback" placeholder="发现设施或服务问题？"></textarea><button @click="submitFeedback">提交反馈</button><p v-if="feedbackStatus" class="success-text">{{ feedbackStatus }}</p><p v-if="feedbackError" class="error-text">{{ feedbackError }}</p></div></aside>
    </main>

    <main v-else-if="active === 'attractions'" class="content"><div class="section-head"><div><p class="eyebrow">EXPLORE / ATTRACTIONS</p><h2>景点浏览</h2></div><span class="count-tag">{{ attractions.length }} 个官方景点</span></div><div v-if="attractionsLoading" class="loading-grid"><div v-for="n in 8" :key="n" class="skeleton-card"></div></div><p v-else-if="attractionsError" class="empty-state error-state">{{ attractionsError }} <button class="secondary" @click="loadAttractions">重新加载</button></p><div v-else class="card-grid"><button v-for="item in attractions" :key="item.attraction_id" class="attraction-card" type="button" @click="openAttraction(item)"><div class="card-image"><img v-if="!imageErrors[item.attraction_id]" :src="imageFor(item)" :alt="`${item.name}景区实景参考图`" loading="lazy" @error="markImageError(item.attraction_id)"><div v-else class="image-fallback"><span>{{ item.valley || '九寨沟' }}</span><strong>{{ item.name }}</strong></div><span class="category-tag">{{ item.category }}</span><span class="view-detail">查看详情 ↗</span></div><div class="card-body"><div class="card-title"><h3>{{ item.name }}</h3><span class="open">{{ item.status === 'open' ? '开放' : '以公告为准' }}</span></div><p>{{ item.description }}</p><div class="meta"><span>⏱ {{ item.visit_duration_minutes }} 分钟</span><span>{{ item.ticket_price === 0 ? '含于景区门票' : `¥ ${item.ticket_price}` }}</span></div></div></button></div></main>

    <main v-else-if="active === 'map'" class="content map-page">
      <section class="map-hero">
        <div class="map-hero-copy">
          <p class="eyebrow">ORIENT / MAP</p>
          <h2>九寨沟三沟地图</h2>
          <p class="map-hero-note">先用地图确认沟谷方向，再安排观光车和栈道顺序。高清原图支持放大查看，适合出发前做路线规划。</p>
          <div class="map-stat-row">
            <div class="map-stat"><strong>3</strong><span>条沟谷</span></div>
            <div class="map-stat"><strong>40+</strong><span>个游览节点</span></div>
            <div class="map-stat"><strong>08:00</strong><span>旺季入园</span></div>
          </div>
          <button class="primary map-hero-cta" @click="active = 'recommend'">去生成路线 ↗</button>
        </div>
        <div class="map-hero-note-card">
          <p class="eyebrow">TRAVEL LOGIC</p>
          <strong>上行乘车，下行游览</strong>
          <p>从诺日朗中心换乘前往高处景点，再沿栈道向下游览，减少折返和爬坡。</p>
        </div>
      </section>

      <section class="map-workspace">
        <div class="map-panel">
          <div class="map-toolbar">
            <div><p class="eyebrow">HIGH RESOLUTION MAP</p><h3>观光车与栈道总览</h3></div>
            <div class="map-controls" aria-label="地图缩放控制">
              <button class="map-control" type="button" aria-label="缩小地图" :disabled="mapScale <= 1" @click="zoomMap(-0.2)">−</button>
              <span>{{ Math.round(mapScale * 100) }}%</span>
              <button class="map-control" type="button" aria-label="放大地图" :disabled="mapScale >= 2.6" @click="zoomMap(0.2)">＋</button>
              <button class="map-control map-reset" type="button" @click="resetMapZoom">1:1</button>
            </div>
          </div>
          <div class="map-viewport">
            <img class="map-image" :src="mapImage" alt="九寨沟三沟观光车与步行栈道高清地图" :style="{ width: `${mapScale * 100}%`, maxWidth: mapScale === 1 ? '100%' : 'none' }">
          </div>
          <p class="map-footnote">拖动地图容器查看细节，放大后可阅读换乘站、栈道和景点标注。</p>
        </div>

        <aside class="map-sidebar">
          <div class="map-legend">
            <p class="eyebrow">MAP LEGEND</p>
            <h3>图例</h3>
            <div class="legend-item"><span class="legend-line route-main"></span><span>观光车主线</span></div>
            <div class="legend-item"><span class="legend-line route-walk"></span><span>步行栈道</span></div>
            <div class="legend-item"><span class="legend-node transfer"></span><span>三沟换乘中心</span></div>
            <div class="legend-item"><span class="legend-node highland"></span><span>高海拔景点提醒</span></div>
          </div>
          <div class="map-sidebar-note">
            <p class="eyebrow">FIELD NOTE</p>
            <strong>诺日朗中心是换乘枢纽</strong>
            <p>日则沟、则查洼沟和树正沟的观光车线路在这里衔接，建议把它作为路线分段的参照点。</p>
          </div>
        </aside>
      </section>

      <section class="valley-section">
        <div class="section-head"><div><p class="eyebrow">THREE VALLEYS</p><h2>三沟怎么走</h2></div><span class="count-tag">按现场开放调整</span></div>
        <div class="valley-grid">
          <article class="valley-card"><span class="valley-index">01</span><div><h3>日则沟</h3><p>原始森林 · 箭竹海 · 熊猫海 · 五花海 · 珍珠滩</p><small>湖泊密度高，适合安排半天的核心景观线。</small></div></article>
          <article class="valley-card"><span class="valley-index">02</span><div><h3>则查洼沟</h3><p>诺日朗中心 · 长海 · 五彩池</p><small>海拔较高，先乘车到长海，再顺路下行。</small></div></article>
          <article class="valley-card"><span class="valley-index">03</span><div><h3>树正沟</h3><p>犀牛海 · 老虎海 · 树正群海 · 芦苇海</p><small>靠近入口和中心区，适合返程时安排慢游。</small></div></article>
        </div>
      </section>

      <section class="map-tips"><div><p class="eyebrow">BEFORE YOU GO</p><h3>出行提示</h3></div><ul><li>先乘车到高处，再按栈道下行，少走回头路。</li><li>长海、五彩池海拔较高，留意体力和天气变化。</li><li>地图用于路线参考，现场以观光车调度和开放公告为准。</li></ul></section>
    </main>

    <main v-else-if="active === 'notices'" class="content"><div class="section-head"><div><p class="eyebrow">OFFICIAL / NOTICES</p><h2>最新景区公告</h2></div><button class="secondary" :disabled="noticesLoading" @click="loadNotices">{{ noticesLoading ? '读取中…' : '刷新公告' }}</button></div><div class="notice-list"><article v-for="item in notices" :key="item.notice_id" class="info-card"><div class="card-title"><h3>{{ item.title }}</h3><span class="live-pill">{{ item.status === 'active' ? '当前有效' : '历史公告' }}</span></div><p>{{ item.content }}</p><small>生效：{{ item.effective_from }} · 来源：九寨沟风景名胜区管理局</small></article><p v-if="!notices.length && !noticesLoading" class="empty-state">暂未读取到公告，请稍后重试或直接咨询智能助手。</p></div></main>

      <main v-else-if="active === 'recommend'" class="content recommend-layout"><section><div class="section-head"><div><p class="eyebrow">PLAN / ROUTE</p><h2>为你安排一条路线</h2></div></div><div class="form-card"><label>游玩时长<select v-model.number="recDuration"><option :value="240">4 小时</option><option :value="480">8 小时</option></select></label><label>同行人群<select v-model="recGroup"><option>普通游客</option><option>老年游客</option><option>亲子家庭</option><option>摄影爱好者</option><option>户外爱好者</option></select></label><label>偏好<select v-model="recPreference"><option>湖泊/海子</option><option>瀑布</option><option>滩流</option><option>森林</option><option>藏寨人文</option></select></label><button class="primary" :disabled="recommendationLoading" @click="recommend">{{ recommendationLoading ? '正在按地图规划…' : '生成推荐路线 ↗' }}</button><p class="form-hint">路线会按日则沟、诺日朗中心、树正沟和则查洼沟的观光车及栈道顺序安排。需要查看高清地图时，请点击顶部“景区地图”。</p></div></section><section v-if="recommendation" class="route-result"><div class="route-summary"><span class="live-pill">{{ recommendation.incomplete ? '需要补充信息' : '可执行建议' }}</span><h3>{{ recommendation.reason }}</h3><p>已安排游览约 {{ recommendation.total_minutes }} 分钟<span v-if="recommendation.time_shortfall_minutes">，还差 {{ recommendation.time_shortfall_minutes }} 分钟</span></p><p class="route-note">{{ recommendation.transport_notes }}</p><p class="route-note">{{ recommendation.rest_notes }}</p></div><div class="route-item" v-for="(item, index) in routeStops" :key="item.attraction_id"><b>{{ String(index + 1).padStart(2, '0') }}</b><div><h3>{{ item.name }}</h3><p>{{ item.valley }} · {{ item.category }} · {{ item.visit_duration_minutes }} 分钟 · {{ item.difficulty }}</p></div></div><div v-if="routeSegments.length" class="segment-list route-segments"><div v-for="(segment, index) in routeSegments" :key="`${segment.from}-${segment.to}-${index}`" class="segment-row"><span>{{ String(index + 1).padStart(2, '0') }}</span><strong>{{ segment.from }} → {{ segment.to }}</strong><small>{{ segment.route_type }}<template v-if="segment.estimated_minutes"> · 约 {{ segment.estimated_minutes }} 分钟</template></small></div></div></section><p v-else-if="recommendationError" class="empty-state error-state">{{ recommendationError }}</p><section v-else class="empty-state">填写条件后生成一条九寨沟观光车 + 步行的常规线路，已依据地图顺序安排节点。</section></main>

    <main v-else class="content admin"><div v-if="!adminToken" class="login-card"><p class="eyebrow">ADMIN ACCESS</p><h2>运营后台</h2><input v-model="adminEmail" placeholder="管理员邮箱"><input v-model="adminPassword" type="password" placeholder="密码"><button class="primary" :disabled="adminLoading" @click="login">{{ adminLoading ? '登录中…' : '登录后台' }}</button><p v-if="adminError" class="error-text">{{ adminError }}</p></div><template v-else><div class="section-head"><div><p class="eyebrow">OPERATIONS / DASHBOARD</p><h2>景区运营概览</h2></div><button class="secondary" :disabled="adminLoading" @click="loadAdmin">{{ adminLoading ? '刷新中…' : '刷新数据' }}</button></div><p v-if="adminError" class="error-banner">{{ adminError }}</p><div v-if="dashboard" class="stat-grid"><div v-for="(value, key) in dashboard.counts" :key="key" class="stat-card"><small>{{ key }}</small><strong>{{ value }}</strong></div></div><section class="admin-table"><div class="table-head"><div><h3>待审核游客反馈</h3><p>采纳后会进入知识文档、Embedding 任务，并在 Milvus 已配置时同步向量索引。</p></div><span>{{ candidates.length }} 条</span></div><p v-if="reviewNotice" class="success-text">{{ reviewNotice }}</p><p v-if="reviewError" class="error-text">{{ reviewError }}</p><div v-if="!candidates.length" class="empty-row">当前没有待审核反馈。</div><div v-for="item in candidates.slice(0, 12)" :key="item.candidate_id" class="table-row"><div><b>{{ item.candidate_id }}</b><p>{{ item.candidate_fact }}</p><small class="status-label">{{ item.status }}</small></div><div class="actions"><button :disabled="reviewBusy === item.candidate_id || item.status !== 'pending_review'" @click="review(item.candidate_id, 'approve')">{{ reviewBusy === item.candidate_id ? '处理中…' : '采纳' }}</button><button class="reject" :disabled="reviewBusy === item.candidate_id || item.status !== 'pending_review'" @click="review(item.candidate_id, 'reject')">驳回</button></div></div></section></template></main>

    <Transition name="fade"><div v-if="selectedAttraction || detailLoading" class="modal-backdrop" @click.self="closeAttraction"><section class="detail-modal" role="dialog" aria-modal="true" aria-label="景点详情"><button class="modal-close" type="button" aria-label="关闭详情" @click="closeAttraction">×</button><div v-if="detailLoading" class="detail-loading"><div class="spinner"></div><p>正在读取景点详情…</p></div><template v-else-if="selectedAttraction"><div class="detail-hero"><img v-if="!imageErrors[selectedAttraction.attraction_id]" :src="imageFor(selectedAttraction)" :alt="`${selectedAttraction.name}景区实景参考图`" @error="markImageError(selectedAttraction.attraction_id)"><div v-else class="image-fallback"><span>{{ selectedAttraction.valley }}</span><strong>{{ selectedAttraction.name }}</strong></div></div><div class="detail-content"><p class="eyebrow">{{ selectedAttraction.valley }} · {{ selectedAttraction.category }}</p><h2>{{ selectedAttraction.name }}</h2><p class="detail-description">{{ selectedAttraction.description }}</p><div class="detail-facts"><span>游览约 {{ selectedAttraction.visit_duration_minutes }} 分钟</span><span>难度 {{ selectedAttraction.difficulty }}</span><span>{{ selectedAttraction.status === 'open' ? '当前开放' : '请以公告为准' }}</span></div><p v-if="selectedAttraction.highlights" class="detail-highlight">看点：{{ selectedAttraction.highlights }}</p><p v-if="selectedAttraction.notice" class="detail-note">游览提示：{{ selectedAttraction.notice }}</p><p v-if="detailError" class="error-text">{{ detailError }}</p><small>信息来源：景区知识库 · 更新时间 {{ selectedAttraction.updated_at?.slice?.(0, 10) || '未提供' }}</small></div></template></section></div></Transition>
  </div>
</template>
