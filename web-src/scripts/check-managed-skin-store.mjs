/** Actual ManagedSkinStore logic; finite authority/CAS/receipt consumer checks. */
import assert from 'node:assert/strict'
import { webcrypto } from 'node:crypto'
import { build } from 'esbuild'
globalThis.crypto ??= webcrypto
const compiled = await build({ entryPoints: ['src/managed-skin-store.ts'], bundle: true, write: false,
  platform: 'node', format: 'esm', target: 'es2022' })
const { ManagedSkinStore } = await import('data:text/javascript;base64,' + Buffer.from(compiled.outputFiles[0].text).toString('base64'))
const schema = 'camera-calibration.preferences.v1'
const value = (skin = 'dark', revision = '0') => ({ skin, version: revision, token: { database_id: 'db-one', schema, revision } })
const receipt = (plan, current = value(plan.skin, String(BigInt(plan.expected.revision) + 1n))) => ({ ...current,
  committed: value(plan.skin, String(BigInt(plan.expected.revision) + 1n)), request_id: plan.request_id, durability: 'sqlite-full' })
const reply = (body, status = 200) => new Response(JSON.stringify(body), { status })
const deferred = () => { let resolve; const promise = new Promise(done => { resolve = done }); return { promise, resolve } }

// Restoration is required; transport failure cannot synthesize a persistent default.
const offline = new ManagedSkinStore(async () => { throw Error('offline') })
await assert.rejects(offline.restore()); assert.throws(() => offline.getSnapshot(), /restore/)
let planOne, count = 0, notified = false
const committed = new ManagedSkinStore(async (path, init) => {
  if (init.method === 'GET') return reply(value())
  count++; planOne = JSON.parse(init.body); return reply(receipt(planOne))
})
await committed.restore(); assert.equal(committed.getSnapshot(), 'dark')
const listener = () => { notified = committed.getSnapshot() === 'light' }
const first = committed.subscribe(listener), second = committed.subscribe(listener)
first(); await committed.setSkin('light'); assert(notified); assert.equal(count, 1); second()
assert.equal(planOne.expected_version, '0'); assert.equal(planOne.expected.revision, '0')

// Keep both rapid choices while the first real write is pending.
const held = deferred(), admitted = deferred(), choices = []
const rapid = new ManagedSkinStore(async (path, init) => {
  if (init.method === 'GET') return reply(value())
  const plan = JSON.parse(init.body); choices.push(plan)
  if (choices.length === 1) { admitted.resolve(); await held.promise }
  return reply(receipt(plan))
})
await rapid.restore(); const light = rapid.setSkin('light'); await admitted.promise
const dark = rapid.setSkin('dark'); held.resolve(); await Promise.all([light, dark])
assert.deepEqual(choices.map(p => [p.skin, p.expected_version]), [['light', '0'], ['dark', '1']])
assert.equal(rapid.getSnapshot(), 'dark')

// Conflict restores actual outside authority and rejects the unsaved choice.
const conflict = new ManagedSkinStore(async (path, init) => init.method === 'PUT'
  ? reply({ error: 'CAS changed', details: { code: 'conflict' } }, 409) : reply(value('dark', '2')))
await conflict.restore(); await assert.rejects(conflict.setSkin('light'), /CAS changed/)
assert.equal(conflict.getSnapshot(), 'dark'); assert.equal(conflict.getIssue(), 'CAS changed')

// Lost reply resolves its original identity; missing receipt never permits a replay/new write.
let lostPlan, batchCount = 0, hasReceipt = false
const unknown = new ManagedSkinStore(async (path, init) => {
  if (path.includes('?request_id=')) {
    assert.equal(new URL(path, 'http://local').searchParams.get('request_id'), lostPlan.request_id)
    return hasReceipt ? reply(receipt(lostPlan)) : reply({ error: 'earlier result unknown', details: { outcome: 'outcome_unknown' } }, 503)
  }
  if (init.method === 'GET') return reply(value())
  batchCount++; const plan = JSON.parse(init.body)
  if (batchCount === 1) { lostPlan = plan; throw Error('reply lost') }
  return reply(receipt(plan))
})
await unknown.restore(); await assert.rejects(unknown.setSkin('light')); assert.equal(unknown.getSnapshot(), 'dark')
await assert.rejects(unknown.setSkin('dark'), /unknown/); assert.equal(batchCount, 1)
hasReceipt = true; await unknown.setSkin('dark'); assert.equal(batchCount, 2); assert.equal(unknown.getSnapshot(), 'dark')

// An outside newer SSE commit wins over the delayed earlier HTTP receipt.
let stateListener, closed = false
const delayed = deferred(), started = deferred()
const outside = new ManagedSkinStore(async (path, init) => {
  if (init.method === 'GET') return reply(value())
  const plan = JSON.parse(init.body); started.resolve(); await delayed.promise; return reply(receipt(plan))
})
await outside.restore(); outside.connectEvents(() => ({ addEventListener(name, listener) { stateListener = listener }, close() { closed = true } }))
const earlier = outside.setSkin('light'); await started.promise
stateListener({ data: JSON.stringify({ preferences: { available: true, snapshot: value('dark', '2') } }) })
delayed.resolve(); await earlier; assert.equal(outside.getSnapshot(), 'dark')
stateListener({ data: JSON.stringify({ preferences: { available: false, snapshot: value('dark', '2'), error: { message: 'storage offline' } } }) })
assert.equal(outside.getSnapshot(), 'dark'); assert.equal(outside.getIssue(), 'storage offline'); outside.close(); assert(closed)

// A 200 without a FULL receipt cannot publish or claim saved.
const invalid = new ManagedSkinStore(async (path, init) => reply(init.method === 'GET' ? value() : value('light', '1')))
await invalid.restore(); await assert.rejects(invalid.setSkin('light'), /FULL receipt/)
assert.equal(invalid.getSnapshot(), 'dark')
console.log('PASS: cold authority, publication before resolve, independent subscribers, rapid choices, CAS, unknown receipt/no replay, outside updates and delayed receipt, invalid FULL')
