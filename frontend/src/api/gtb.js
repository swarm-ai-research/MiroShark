import service from './index'

export const startGtb = (simId, body = {}) =>
  service.post(`/api/gtb/${simId}/start`, body)

export const stepGtb = (simId, n = 1) =>
  service.post(`/api/gtb/${simId}/step`, { n })

export const getGtbState = (simId) =>
  service.get(`/api/gtb/${simId}/state`)

export const stopGtb = (simId) =>
  service.post(`/api/gtb/${simId}/stop`)

export const listGtbWorlds = () =>
  service.get('/api/gtb/')

export const getGtbMarkets = (simId) =>
  service.get(`/api/gtb/${simId}/markets`)

export const generateGtbMarkets = (simId) =>
  service.post(`/api/gtb/${simId}/markets/generate`)

export const getGtbPolymarket = (simId) =>
  service.get(`/api/gtb/${simId}/polymarket`)
