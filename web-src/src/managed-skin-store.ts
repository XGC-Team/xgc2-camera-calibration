import type { SkinStore, XGCSkin } from '@xgc2/ui-react'

type Token = { database_id: string; schema: string; revision: string }
type Snapshot = { skin: XGCSkin; version: string; token: Token }
const schema = 'camera-calibration.preferences.v1'
const decimal = /^(0|[1-9][0-9]{0,19})$/
const identity = /^[A-Za-z0-9._-]{1,128}$/

class PreferenceFailure extends Error {
  constructor(message: string, readonly code = 'unavailable', readonly uncertain = false) { super(message) }
}
function snapshot(value: unknown): Snapshot {
  const item = value as Partial<Snapshot> | null
  if (!item || !['light', 'dark'].includes(item.skin || '') || !decimal.test(item.version || '')
      || !item.token || !identity.test(item.token.database_id || '') || item.token.schema !== schema
      || !decimal.test(item.token.revision || '') || BigInt(item.version!) > BigInt(item.token.revision)
      || BigInt(item.token.revision) > 9223372036854775807n) {
    throw new PreferenceFailure('Preference storage returned an invalid snapshot', 'invalid_response')
  }
  return { skin: item.skin!, version: item.version!, token: { ...item.token } }
}

/** One authority per mounted application, restored from its granted Storage scope. */
export class ManagedSkinStore implements SkinStore {
  private current: Snapshot | undefined
  private listeners = new Set<() => void>()
  private tail: Promise<void> = Promise.resolve()
  private pending: string | undefined
  private events: EventSource | undefined
  private issue: string | undefined
  constructor(private readonly request: typeof fetch = fetch) {}

  getSnapshot = (): XGCSkin => {
    if (!this.current) throw new Error('Preference storage must restore before initializeSkin')
    return this.current.skin
  }
  getIssue = () => this.issue
  subscribe = (listener: () => void) => {
    const registration = () => listener()
    this.listeners.add(registration)
    return () => { this.listeners.delete(registration) }
  }
  private notify() { for (const listener of [...this.listeners]) listener() }
  private publish(value: Snapshot) {
    if (this.current) {
      if (value.token.database_id !== this.current.token.database_id) {
        throw new PreferenceFailure('Preference database binding changed; reload required', 'conflict')
      }
      if (BigInt(value.token.revision) < BigInt(this.current.token.revision)) return
      if (BigInt(value.version) < BigInt(this.current.version)
          || value.version === this.current.version && value.skin !== this.current.skin) {
        throw new PreferenceFailure('Preference record changed without a new version', 'invalid_response')
      }
    }
    this.current = value
    this.issue = undefined
    this.notify()
  }
  private async call(path: string, body?: unknown): Promise<Record<string, unknown>> {
    const abort = new AbortController()
    const deadline = setTimeout(() => abort.abort(), 12000)
    try {
      const response = await this.request(path, { method: body === undefined ? 'GET' : 'PUT',
        headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
        body: body === undefined ? undefined : JSON.stringify(body), signal: abort.signal, cache: 'no-store' })
      let result: Record<string, unknown>
      try { result = await response.json() as Record<string, unknown> }
      catch { throw new PreferenceFailure('Preference response is invalid', 'invalid_response', body !== undefined) }
      if (!response.ok) {
        const details = result?.details as { code?: string; outcome?: string } | undefined
        throw new PreferenceFailure(typeof result?.error === 'string' ? result.error : 'Preference storage is unavailable',
          details?.code || (response.status === 409 ? 'conflict' : 'unavailable'), details?.outcome === 'outcome_unknown')
      }
      return result
    } catch (error) {
      if (error instanceof PreferenceFailure) throw error
      throw new PreferenceFailure('Preference call did not complete', 'unavailable', body !== undefined)
    } finally { clearTimeout(deadline) }
  }
  async restore() {
    try { this.publish(snapshot(await this.call('api/v1/preferences'))) }
    catch (error) { this.issue = String((error as Error).message); this.notify(); throw error }
  }
  private async resolvePending() {
    if (!this.pending) return
    const result = await this.call('api/v1/preferences?request_id=' + encodeURIComponent(this.pending))
    if (result.request_id !== this.pending || result.durability !== 'sqlite-full') {
      throw new PreferenceFailure('Earlier appearance write has no FULL receipt', 'outcome_unknown', true)
    }
    snapshot(result.committed)
    this.publish(snapshot(result))
    this.pending = undefined
  }
  setSkin = (next: XGCSkin): Promise<void> => {
    const work = this.tail.catch(() => undefined).then(async () => {
      let requestId: string | undefined
      try {
        await this.resolvePending()
        if (!this.current) throw new PreferenceFailure('Preference storage has not restored')
        if (this.current.skin === next) return
        requestId = crypto.randomUUID()
        const expected = this.current
        const result = await this.call('api/v1/preferences', { skin: next, expected_version: this.current.version,
          expected: this.current.token, request_id: requestId })
        if (result.request_id !== requestId || result.durability !== 'sqlite-full') {
          throw new PreferenceFailure('Appearance write has no FULL receipt', 'outcome_unknown', true)
        }
        let current: Snapshot
        try {
          const committed = snapshot(result.committed)
          if (committed.skin !== next || committed.token.database_id !== expected.token.database_id
              || BigInt(committed.version) !== BigInt(expected.token.revision) + 1n
              || committed.version !== committed.token.revision) throw new Error('Receipt differs from the appearance plan')
          current = snapshot(result)
        } catch { throw new PreferenceFailure('Appearance receipt is invalid', 'outcome_unknown', true) }
        this.publish(current) // Publish committed authority before resolving useSkin's promise.
      } catch (error) {
        if (requestId && (error as PreferenceFailure).uncertain) this.pending = requestId
        if ((error as PreferenceFailure).code === 'conflict') {
          try { await this.restore() } catch { /* Preserve the actual failure and committed snapshot. */ }
        }
        this.issue = (error as Error).message
        this.notify()
        throw error
      }
    })
    this.tail = work
    return work
  }
  connectEvents(create: (url: string) => EventSource = url => new EventSource(url)) {
    if (this.events) return
    this.events = create('api/v1/events')
    this.events.addEventListener('state', event => {
      try {
        const preferences = JSON.parse((event as MessageEvent<string>).data).preferences
        if (!preferences?.available) throw new PreferenceFailure(preferences?.error?.message || 'Preference storage is unavailable')
        this.publish(snapshot(preferences.snapshot))
      } catch (error) { this.issue = (error as Error).message; this.notify() }
    })
    this.events.onerror = () => { this.issue = 'Preference updates disconnected; changes still require storage confirmation'; this.notify() }
  }
  close() { this.events?.close(); this.events = undefined }
}
