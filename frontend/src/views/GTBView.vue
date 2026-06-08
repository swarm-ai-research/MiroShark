<template>
  <div class="gtb">
    <header class="gtb-header">
      <h1>Gather · Trade · Build</h1>
      <div class="controls">
        <input v-model="simId" placeholder="sim id" class="sim-input" />
        <button @click="onStart" :disabled="busy">Start</button>
        <button @click="onStep(1)" :disabled="busy || !state">Step</button>
        <button @click="onStep(5)" :disabled="busy || !state">Step ×5</button>
        <button @click="onGenerateMarkets" :disabled="busy || !state">Seed Markets</button>
        <label class="toggle">
          <input type="checkbox" v-model="autoPoll" :disabled="!state" />
          auto&nbsp;{{ pollMs }}ms
        </label>
        <button @click="onStop" :disabled="busy || !state" class="danger">Stop</button>
      </div>
      <div v-if="error" class="error">{{ error }}</div>
    </header>

    <div v-if="state" class="layout">
      <section class="grid-panel">
        <h2>Grid · epoch {{ state.epoch }} · step {{ state.step_in_epoch }}</h2>
        <svg
          :viewBox="`0 0 ${state.config.map.width * cell} ${state.config.map.height * cell}`"
          class="grid"
          preserveAspectRatio="xMidYMid meet"
        >
          <rect
            v-for="res in state.resources"
            :key="`res-${res.position[0]}-${res.position[1]}`"
            :x="res.position[1] * cell"
            :y="res.position[0] * cell"
            :width="cell" :height="cell"
            :fill="resourceColor(res.type)"
            :opacity="0.4 + Math.min(1, res.amount / 5) * 0.5"
          />
          <rect
            v-for="h in state.houses"
            :key="`h-${h.position[0]}-${h.position[1]}-${h.owner_id}`"
            :x="h.position[1] * cell + 1"
            :y="h.position[0] * cell + 1"
            :width="cell - 2" :height="cell - 2"
            fill="none" stroke="#f9a826" stroke-width="2"
          />
          <circle
            v-for="w in state.workers"
            :key="`w-${w.agent_id}`"
            :cx="w.position[1] * cell + cell / 2"
            :cy="w.position[0] * cell + cell / 2"
            :r="cell / 3"
            :fill="workerColor(w)"
            :stroke="w.times_caught > 0 ? '#ff5577' : '#ffffff'"
            stroke-width="1"
          >
            <title>{{ w.agent_id }} · {{ w.policy }} · coin {{ w.inventory.coin?.toFixed(1) }} · wood {{ w.inventory.wood?.toFixed(1) }} · stone {{ w.inventory.stone?.toFixed(1) }}</title>
          </circle>
        </svg>
        <div class="legend">
          <span class="dot wood"></span> wood
          <span class="dot stone"></span> stone
          <span class="dot honest"></span> honest
          <span class="dot llm"></span> llm
          <span class="dot gaming"></span> gaming
          <span class="dot evasive"></span> evasive
          <span class="dot collusive"></span> collusive
          <span class="legend-house"></span> house
        </div>
      </section>

      <aside class="side-panel">
        <section v-if="headline" class="headline-card" :class="`dir-${headline.direction.toLowerCase()}`">
          <div class="headline-row">
            <span class="badge">Top signal</span>
            <span class="dir">{{ headline.direction }}</span>
            <span class="tier">{{ headline.confidence_tier }}</span>
          </div>
          <div class="q">{{ headline.suggested_market_title }}</div>
          <div class="probs">
            <span class="yes">YES {{ (headline.yes_probability * 100).toFixed(1) }}%</span>
            <span class="sep">·</span>
            <span class="no">NO {{ (headline.no_probability * 100).toFixed(1) }}%</span>
            <span class="sep">·</span>
            <span class="conf">conf {{ headline.confidence_pct.toFixed(0) }}</span>
          </div>
          <div class="meta">
            pools YES {{ headline.yes_pool.toFixed(1) }} / NO {{ headline.no_pool.toFixed(1) }}
            · deadline ep {{ headline.deadline_epoch }}
          </div>
        </section>

        <section>
          <h3>Tax brackets</h3>
          <table class="brackets">
            <thead><tr><th>≥ income</th><th>rate</th></tr></thead>
            <tbody>
              <tr v-for="(b, i) in state.tax_brackets" :key="i">
                <td>{{ b.threshold.toFixed(2) }}</td>
                <td>{{ (b.rate * 100).toFixed(1) }}%</td>
              </tr>
            </tbody>
          </table>
        </section>

        <section v-if="state.last_metrics">
          <h3>Last epoch metrics</h3>
          <ul class="metric-list">
            <li><span>welfare</span><span>{{ state.last_metrics.welfare.toFixed(3) }}</span></li>
            <li><span>gini</span><span>{{ state.last_metrics.gini_coefficient.toFixed(3) }}</span></li>
            <li><span>production</span><span>{{ state.last_metrics.total_production.toFixed(2) }}</span></li>
            <li><span>tax revenue</span><span>{{ state.last_metrics.total_tax_revenue.toFixed(2) }}</span></li>
            <li><span>audits / catches</span><span>{{ state.last_metrics.total_audits }} / {{ state.last_metrics.total_catches }}</span></li>
            <li><span>bunching</span><span>{{ state.last_metrics.bunching_intensity.toFixed(3) }}</span></li>
          </ul>
        </section>

        <section v-if="state.markets">
          <h3>Markets ({{ state.markets.open.length }} open · {{ state.markets.resolved.length }} resolved)</h3>
          <div class="market-list">
            <div v-for="m in state.markets.open" :key="m.market_id" class="market open">
              <div class="q">{{ m.question }}</div>
              <div class="meta">
                deadline ep {{ m.deadline_epoch }} · created ep {{ m.created_epoch }}
                <span v-if="marketStakeTotals[m.market_id]" class="stake-tag">
                  · YES {{ marketStakeTotals[m.market_id].yes.toFixed(1) }} / NO {{ marketStakeTotals[m.market_id].no.toFixed(1) }}
                </span>
              </div>
            </div>
            <div v-for="m in lastResolved" :key="m.market_id" class="market" :class="m.status">
              <div class="q">{{ m.question }}</div>
              <div class="meta">{{ m.status.toUpperCase() }} at ep {{ m.resolved_epoch }} (value {{ m.resolved_value?.toFixed(2) ?? '—' }})</div>
            </div>
          </div>
        </section>

        <section v-if="state.stakes">
          <h3>Open stakes ({{ openStakeCount }})</h3>
          <div v-if="openStakeCount === 0" class="muted">No agents have positions on open markets.</div>
          <table v-else class="stakes">
            <thead><tr><th>agent</th><th>market</th><th>side</th><th>amt</th></tr></thead>
            <tbody>
              <tr v-for="s in openStakeRows" :key="`${s.agent_id}-${s.market_id}-${s.epoch}`">
                <td>{{ s.agent_id }}</td>
                <td class="market-id">{{ s.market_id }}</td>
                <td :class="['side', s.side]">{{ s.side.toUpperCase() }}</td>
                <td>{{ s.amount.toFixed(2) }}</td>
              </tr>
            </tbody>
          </table>
          <details v-if="recentStakeHistory.length" class="history">
            <summary>Recent stake events ({{ recentStakeHistory.length }})</summary>
            <ul>
              <li v-for="(h, i) in recentStakeHistory" :key="i" :class="['hist', h.event]">
                <code>{{ h.event }}</code> {{ h.agent_id }} · {{ h.market_id }}
                <span v-if="h.side"> · {{ h.side.toUpperCase() }} {{ h.amount?.toFixed?.(2) }}</span>
                <span v-if="h.gross_payout"> → +{{ h.gross_payout.toFixed(2) }}</span>
              </li>
            </ul>
          </details>
        </section>
      </aside>

      <section v-if="state.metrics_history.length" class="charts">
        <h3>Metrics over epochs</h3>
        <div class="chart-row">
          <Sparkline label="welfare"   :values="series('welfare')" stroke="#7cf29c" />
          <Sparkline label="gini"      :values="series('gini_coefficient')" stroke="#f9a826" />
          <Sparkline label="prod"      :values="series('total_production')" stroke="#7ab8ff" />
          <Sparkline label="tax"       :values="series('total_tax_revenue')" stroke="#ff7ad9" />
          <Sparkline label="bunching"  :values="series('bunching_intensity')" stroke="#c084fc" />
        </div>
      </section>
    </div>

    <p v-else class="empty">No world loaded. Pick a sim id and hit Start.</p>
  </div>
</template>

<script setup>
import { ref, computed, watch, onBeforeUnmount } from 'vue'
import {
  startGtb, stepGtb, getGtbState, stopGtb,
  generateGtbMarkets, getGtbPolymarket,
} from '../api/gtb'
import Sparkline from '../components/GtbSparkline.vue'

const props = defineProps({ simId: { type: String, default: '' } })
const simId = ref(props.simId || 'gtb-demo')
const state = ref(null)
const polymarket = ref(null)
const error = ref(null)
const busy = ref(false)
const cell = 24
const autoPoll = ref(false)
const pollMs = 1500
let pollTimer = null

async function refresh () {
  const [s, p] = await Promise.all([
    getGtbState(simId.value),
    getGtbPolymarket(simId.value).catch(() => null),
  ])
  state.value = s.data.state
  polymarket.value = p?.data || null
}

async function withBusy (fn) {
  busy.value = true
  error.value = null
  try { await fn() } catch (e) {
    error.value = e?.response?.data?.error || e?.message || String(e)
  } finally { busy.value = false }
}

const onStart = () => withBusy(async () => {
  const r = await startGtb(simId.value, {})
  state.value = r.data.state
  await refresh()
})

const onStep = (n) => withBusy(async () => {
  await stepGtb(simId.value, n)
  await refresh()
})

const onStop = () => withBusy(async () => {
  await stopGtb(simId.value)
  state.value = null
  polymarket.value = null
})

const onGenerateMarkets = () => withBusy(async () => {
  await generateGtbMarkets(simId.value)
  await refresh()
})

const lastResolved = computed(() => {
  if (!state.value) return []
  return state.value.markets.resolved.slice(-5).reverse()
})

const openStakeRows = computed(() => {
  const book = state.value?.stakes?.open_stakes || {}
  const rows = []
  for (const mid of Object.keys(book)) for (const s of book[mid]) rows.push(s)
  return rows.sort((a, b) => a.agent_id.localeCompare(b.agent_id))
})

const openStakeCount = computed(() => openStakeRows.value.length)

const marketStakeTotals = computed(() => {
  const totals = {}
  for (const s of openStakeRows.value) {
    if (!totals[s.market_id]) totals[s.market_id] = { yes: 0, no: 0 }
    totals[s.market_id][s.side] = (totals[s.market_id][s.side] || 0) + s.amount
  }
  return totals
})

const recentStakeHistory = computed(() =>
  (state.value?.stakes?.history || []).slice(-10).reverse()
)

const headline = computed(() => polymarket.value?.headline || null)

const series = (key) =>
  (state.value?.metrics_history || []).map(m => m[key])

function stopPoll () { if (pollTimer) { clearInterval(pollTimer); pollTimer = null } }
function startPoll () {
  stopPoll()
  pollTimer = setInterval(async () => {
    if (busy.value || !state.value) return
    try { await refresh() } catch (e) { /* tolerate transient errors silently */ }
  }, pollMs)
}
watch(autoPoll, (on) => { on ? startPoll() : stopPoll() })
watch(state, (s) => { if (!s) { autoPoll.value = false; stopPoll() } })
onBeforeUnmount(stopPoll)

const resourceColor = (t) => t === 'wood' ? '#3aa55c' : t === 'stone' ? '#8a93a1' : '#f9a826'

const POLICY_COLOR = {
  HonestWorkerPolicy: '#7ab8ff',
  LLMWorkerPolicy: '#a78bfa',
  GamingWorkerPolicy: '#f9a826',
  EvasiveWorkerPolicy: '#ff7ad9',
  CollusiveWorkerPolicy: '#ff5577',
}
const workerColor = (w) => POLICY_COLOR[w.policy] || '#cccccc'
</script>

<style scoped>
.gtb { padding: 16px 24px; color: #e6edf3; background: #010409; min-height: 100vh; }
.gtb-header h1 { margin: 0 0 8px 0; font-size: 22px; }
.controls { display: flex; gap: 8px; align-items: center; margin-bottom: 8px; }
.sim-input { padding: 6px 10px; border-radius: 4px; border: 1px solid #30363d; background: #0d1117; color: #e6edf3; }
button { padding: 6px 12px; border-radius: 4px; border: 1px solid #30363d; background: #161b22; color: #e6edf3; cursor: pointer; }
button:disabled { opacity: 0.4; cursor: not-allowed; }
button.danger { border-color: #ff5577; color: #ff5577; }
.error { color: #ff5577; margin-top: 6px; font-family: monospace; font-size: 13px; }
.layout { display: grid; grid-template-columns: minmax(420px, 1fr) 360px; grid-template-rows: auto auto; gap: 16px; margin-top: 12px; }
.grid-panel { grid-row: 1 / 2; background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 12px; }
.grid-panel h2 { margin: 0 0 8px 0; font-size: 14px; font-weight: 500; color: #8b949e; }
.grid { width: 100%; height: auto; background: #010409; border: 1px solid #161b22; }
.legend { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; font-size: 12px; color: #8b949e; margin-top: 8px; }
.legend .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }
.legend .dot.wood { background: #3aa55c; }
.legend .dot.stone { background: #8a93a1; }
.legend .dot.honest { background: #7ab8ff; }
.legend .dot.llm { background: #a78bfa; }
.legend .dot.gaming { background: #f9a826; }
.legend .dot.evasive { background: #ff7ad9; }
.legend .dot.collusive { background: #ff5577; }
.legend-house { display: inline-block; width: 10px; height: 10px; border: 2px solid #f9a826; margin-right: 4px; vertical-align: middle; }
.side-panel { grid-row: 1 / 3; display: flex; flex-direction: column; gap: 12px; }
.side-panel section { background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 12px; }
.side-panel h3 { margin: 0 0 8px 0; font-size: 13px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.04em; }
.brackets { width: 100%; font-family: monospace; font-size: 13px; }
.brackets th { text-align: left; color: #8b949e; font-weight: 400; padding-bottom: 4px; }
.brackets td { padding: 2px 0; }
.metric-list { list-style: none; padding: 0; margin: 0; font-family: monospace; font-size: 13px; }
.metric-list li { display: flex; justify-content: space-between; padding: 2px 0; border-bottom: 1px dotted #21262d; }
.metric-list li:last-child { border-bottom: 0; }
.market-list { display: flex; flex-direction: column; gap: 6px; max-height: 320px; overflow-y: auto; }
.market { padding: 6px 8px; border-left: 3px solid #30363d; background: #010409; font-size: 12px; border-radius: 0 4px 4px 0; }
.market.open { border-left-color: #7ab8ff; }
.market.yes { border-left-color: #7cf29c; opacity: 0.85; }
.market.no { border-left-color: #ff5577; opacity: 0.85; }
.market.expired { border-left-color: #8b949e; opacity: 0.6; }
.market .q { color: #e6edf3; }
.market .meta { color: #8b949e; font-family: monospace; font-size: 11px; margin-top: 2px; }
.charts { grid-row: 2 / 3; background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 12px; }
.charts h3 { margin: 0 0 8px 0; font-size: 13px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.04em; }
.chart-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
.empty { color: #8b949e; margin-top: 24px; }
.toggle { display: inline-flex; align-items: center; gap: 4px; color: #8b949e; font-size: 12px; user-select: none; }
.toggle input { accent-color: #7ab8ff; }
.stake-tag { color: #c084fc; font-family: monospace; }
.muted { color: #8b949e; font-size: 12px; }
.stakes { width: 100%; font-family: monospace; font-size: 12px; border-collapse: collapse; }
.stakes th { text-align: left; color: #8b949e; font-weight: 400; padding-bottom: 4px; border-bottom: 1px solid #21262d; }
.stakes td { padding: 3px 4px; border-bottom: 1px dotted #21262d; }
.stakes td.side { font-weight: 600; }
.stakes td.side.yes { color: #7cf29c; }
.stakes td.side.no { color: #ff7ad9; }
.stakes td.market-id { color: #8b949e; }
.history { margin-top: 8px; }
.history summary { color: #8b949e; cursor: pointer; font-size: 12px; }
.history ul { list-style: none; padding: 6px 0 0 0; margin: 0; max-height: 180px; overflow-y: auto; }
.history li { font-family: monospace; font-size: 11px; color: #e6edf3; padding: 2px 0; }
.history li code { color: #8b949e; }
.history li.won { color: #7cf29c; }
.history li.lost { color: #ff7ad9; }
.history li.stake_rejected { color: #ff5577; }
.headline-card { border-left: 4px solid #30363d; }
.headline-card.dir-bullish { border-left-color: #7cf29c; }
.headline-card.dir-bearish { border-left-color: #ff7ad9; }
.headline-card.dir-neutral { border-left-color: #8b949e; }
.headline-card .headline-row { display: flex; gap: 8px; align-items: center; margin-bottom: 6px; }
.headline-card .badge { font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; background: #21262d; color: #8b949e; padding: 2px 6px; border-radius: 3px; }
.headline-card .dir { font-weight: 600; font-size: 13px; }
.headline-card.dir-bullish .dir { color: #7cf29c; }
.headline-card.dir-bearish .dir { color: #ff7ad9; }
.headline-card.dir-neutral .dir { color: #8b949e; }
.headline-card .tier { font-size: 11px; color: #c084fc; text-transform: uppercase; letter-spacing: 0.04em; }
.headline-card .q { color: #e6edf3; font-size: 13px; margin-bottom: 6px; }
.headline-card .probs { font-family: monospace; font-size: 12px; display: flex; gap: 6px; align-items: center; }
.headline-card .probs .yes { color: #7cf29c; }
.headline-card .probs .no { color: #ff7ad9; }
.headline-card .probs .conf { color: #c084fc; }
.headline-card .probs .sep { color: #30363d; }
.headline-card .meta { color: #8b949e; font-family: monospace; font-size: 11px; margin-top: 4px; }
</style>
