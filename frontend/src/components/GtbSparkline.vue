<template>
  <div class="sparkline">
    <div class="sparkline-header">
      <span class="label">{{ label }}</span>
      <span class="value">{{ last?.toFixed?.(3) ?? '—' }}</span>
    </div>
    <svg viewBox="0 0 100 30" preserveAspectRatio="none" class="sparkline-svg">
      <polyline
        v-if="points"
        :points="points"
        fill="none"
        :stroke="stroke"
        stroke-width="1.5"
      />
      <text v-else x="50" y="18" text-anchor="middle" fill="#30363d" font-size="8">no data</text>
    </svg>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  label: { type: String, required: true },
  values: { type: Array, default: () => [] },
  stroke: { type: String, default: '#7ab8ff' },
})

const last = computed(() => props.values.length ? props.values[props.values.length - 1] : null)

const points = computed(() => {
  const vs = props.values.filter(v => typeof v === 'number' && !Number.isNaN(v))
  if (vs.length < 2) return null
  const min = Math.min(...vs)
  const max = Math.max(...vs)
  const range = max - min || 1
  const stepX = 100 / (vs.length - 1)
  return vs.map((v, i) => `${(i * stepX).toFixed(2)},${(28 - ((v - min) / range) * 26).toFixed(2)}`).join(' ')
})
</script>

<style scoped>
.sparkline { background: #010409; border: 1px solid #161b22; border-radius: 4px; padding: 6px 8px; }
.sparkline-header { display: flex; justify-content: space-between; font-family: monospace; font-size: 11px; color: #8b949e; margin-bottom: 2px; }
.sparkline-header .value { color: #e6edf3; }
.sparkline-svg { width: 100%; height: 36px; display: block; }
</style>
