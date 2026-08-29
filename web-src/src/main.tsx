import { useEffect, useRef, useState, type RefObject } from 'react'
import { flushSync } from 'react-dom'
import { createRoot } from 'react-dom/client'
import {
  AppShell,
  Button,
  CodeBlock,
  DataTable,
  DescriptionItem,
  DescriptionList,
  FormField,
  initializeSkin,
  Notice,
  Panel,
  ProductBrand,
  ResourceMeter,
  ResponsiveSplit,
  ScrollRegion,
  SectionHeader,
  SegmentedControl,
  Select,
  StatusText,
  Topbar,
  useSkin,
} from '@xgc2/ui-react'
import '@xgc2/ui-react/styles.css'
import './styles.css'

type StatusTone = 'neutral' | 'info' | 'success' | 'warning' | 'danger'

const SKIN_STORAGE_KEY = 'xgc2-camera-calibration.skin'
initializeSkin({ defaultSkin: 'dark', storageKey: SKIN_STORAGE_KEY })

function useLegacyMutation(ref: RefObject<HTMLElement | null>, sync: () => void) {
  useEffect(() => {
    const node = ref.current
    if (!node) return
    sync()
    const observer = new MutationObserver(sync)
    observer.observe(node, { attributes: true, childList: true, subtree: true })
    return () => observer.disconnect()
  }, [])
}

function toneForStatus(value: string, className: string): StatusTone {
  const normalized = value.trim().toLowerCase()
  if (/disconnect|unavailable|failed|error/.test(normalized)) return 'danger'
  if (/connect|wait|live/.test(normalized)) return 'info'
  if (/frozen|saved|complete/.test(normalized) || /pill-on/.test(className)) return 'success'
  return className.includes('muted') ? 'neutral' : 'info'
}

function LegacyStatus({
  id,
  initial,
  sourceClassName = '',
  hideValues = [],
}: {
  id: string
  initial: string
  sourceClassName?: string
  hideValues?: string[]
}) {
  const ref = useRef<HTMLSpanElement>(null)
  const [snapshot, setSnapshot] = useState({ className: sourceClassName, value: initial })
  const sync = () => {
    const node = ref.current
    if (node) setSnapshot({ className: node.className, value: node.textContent || '' })
  }
  useLegacyMutation(ref, sync)
  const hidden = hideValues.includes(snapshot.value.trim().toLowerCase())
  return (
    <>
      <span ref={ref} id={id} className={`legacy-state-source ${sourceClassName}`}>{initial}</span>
      {hidden ? null : (
        <StatusText status={snapshot.value || 'idle'} tone={toneForStatus(snapshot.value, snapshot.className)}>
          {snapshot.value}
        </StatusText>
      )}
    </>
  )
}

function LegacyNotice({ id, tone }: { id: string; tone: 'danger' | 'success' }) {
  const ref = useRef<HTMLElement>(null)
  const [snapshot, setSnapshot] = useState({ visible: false, value: '' })
  const sync = () => {
    const node = ref.current
    if (node) setSnapshot({ visible: !node.classList.contains('hidden') && Boolean(node.textContent), value: node.textContent || '' })
  }
  useLegacyMutation(ref, sync)
  return (
    <>
      <section ref={ref} id={id} className="legacy-state-source hidden" aria-hidden="true" />
      {snapshot.visible ? <Notice density="compact" role={tone === 'danger' ? 'alert' : 'status'} tone={tone}>{snapshot.value}</Notice> : null}
    </>
  )
}

function LegacyCodeResult({ id, initial = '', initiallyHidden = false }: { id: string; initial?: string; initiallyHidden?: boolean }) {
  const ref = useRef<HTMLPreElement>(null)
  const [snapshot, setSnapshot] = useState({ visible: !initiallyHidden, value: initial })
  const sync = () => {
    const node = ref.current
    if (node) setSnapshot({ visible: !node.hidden && Boolean(node.textContent), value: node.textContent || '' })
  }
  useLegacyMutation(ref, sync)
  return (
    <>
      <pre ref={ref} id={id} className="legacy-state-source" hidden={initiallyHidden}>{initial}</pre>
      {snapshot.visible ? <CodeBlock className="calibration-result" content={snapshot.value} label="Result" language="text" /> : null}
    </>
  )
}

type Meter = { label: string; percent: number }

function LegacyCoverage() {
  const ref = useRef<HTMLDivElement>(null)
  const [meters, setMeters] = useState<Meter[]>([])
  const sync = () => {
    const node = ref.current
    if (!node) return
    setMeters(Array.from(node.querySelectorAll<HTMLElement>('.bar')).map((bar) => ({
      label: bar.querySelector('.bar-label span')?.textContent || 'Coverage',
      percent: Number.parseFloat(bar.querySelector('.pct')?.textContent || '0') || 0,
    })))
  }
  useLegacyMutation(ref, sync)
  return (
    <>
      <div ref={ref} id="bars" className="legacy-state-source" />
      <div className="calibration-meters">
        {meters.map((meter) => (
          <ResourceMeter
            key={meter.label}
            label={meter.label}
            detail={`${meter.percent}%`}
            percent={meter.percent}
            tone={meter.percent >= 100 ? 'success' : 'warning'}
          />
        ))}
      </div>
    </>
  )
}

function ThemeControl() {
  const [skin, setSkin] = useSkin({ defaultSkin: 'dark', storageKey: SKIN_STORAGE_KEY })
  return (
    <SegmentedControl
      ariaLabel="Appearance"
      value={skin}
      options={[{ label: 'Light', value: 'light' }, { label: 'Dark', value: 'dark' }]}
      onValueChange={(value) => setSkin(value === 'light' ? 'light' : 'dark')}
    />
  )
}

function IntrinsicPage() {
  return (
    <AppShell
      className="calibration-shell"
      contentClassName="calibration-content"
      contentPadding="none"
      mobileBreakpoint="compact"
      mobileLayout="document"
      topbar={<Topbar brand={<ProductBrand product="Camera intrinsic calibration" />} actions={<ThemeControl />} />}
    >
      <div className="calibration-page calibration-page-intrinsic">
        <ResponsiveSplit
          primary={(
            <Panel
              bodyLayout="column"
              className="calibration-view-panel"
              fill
              padding="none"
              title="Live camera"
              actions={<LegacyStatus id="conn" initial="connecting…" sourceClassName="pill pill-off" hideValues={['connected']} />}
            >
              <div className="calibration-frame"><img id="stream" alt="Camera stream" /></div>
              <p className="calibration-hint">
                Move the camera so the board is seen near and far, at the image edges and tilted, until all four bars fill; then <strong>Calibrate</strong>. In simulation, click a sphere in the guide or use Auto-run.
              </p>
            </Panel>
          )}
          secondary={(
            <ScrollRegion className="calibration-side" fill>
              <Panel title="Board detection" actions={<LegacyStatus id="detection-status" initial="waiting" sourceClassName="pill pill-off" />}>
                <div className="calibration-detection-summary">
                  <strong id="detection-corners">No detection result yet</strong>
                  <span id="detection-frame" className="calibration-meta">No captured frame</span>
                </div>
                <div id="detection-metrics" className="calibration-detection-metrics" />
                <p id="detection-sample" className="calibration-meta">Capture a frame to run board detection.</p>
              </Panel>

              <Panel title="Coverage" actions={<span id="samples" className="calibration-meta">0 samples</span>}>
                <LegacyCoverage />
              </Panel>

              <Panel title="Calibration">
                <div className="calibration-actions">
                  <Button id="btn-calibrate" className="calibration-action" tone="primary" appearance="solid" disabled>Calibrate and save</Button>
                  <Button id="btn-reset" className="calibration-action">Reset</Button>
                </div>
                <LegacyStatus id="status" initial="" hideValues={['']} />
                <LegacyCodeResult id="result" initiallyHidden />
              </Panel>

              <Panel id="camera-card" title="Sample guide">
                <p className="calibration-hint">Drag to rotate the guide; select a sphere to move the simulated camera.</p>
                <canvas id="scene" />
                <div className="calibration-next">
                  <p>Next: <strong id="next-name">—</strong><span id="done-count" className="calibration-meta" /></p>
                  <img id="ref-img" alt="Expected view" hidden />
                  <p id="ref-hint" className="calibration-meta">Move the camera to fill coverage; captured poses are marked in the guide.</p>
                </div>
                <div className="calibration-actions">
                  <Button id="btn-reset-pose" className="calibration-action" disabled>Reset pose</Button>
                  <Button id="btn-auto" className="calibration-action" disabled>Auto-run</Button>
                </div>
                <p id="pose" className="calibration-meta calibration-pose" />
              </Panel>
            </ScrollRegion>
          )}
        />
      </div>
    </AppShell>
  )
}

function ExtrinsicPage() {
  return (
    <AppShell
      className="calibration-shell"
      contentClassName="calibration-content"
      contentPadding="none"
      mobileBreakpoint="compact"
      mobileLayout="document"
      topbar={<Topbar brand={<ProductBrand product="Camera extrinsic calibration" />} actions={<ThemeControl />} />}
    >
      <div className="calibration-page calibration-page-extrinsic">
        <div className="calibration-notices">
          <LegacyNotice id="error-banner" tone="danger" />
          <LegacyNotice id="success-banner" tone="success" />
        </div>
        <ResponsiveSplit
          className="calibration-workspace"
          primary={(
            <Panel
              bodyLayout="column"
              className="calibration-view-panel"
              fill
              padding="none"
              title="Camera frame"
              actions={(
                <div className="calibration-health">
                  <LegacyStatus id="mode-chip" initial="Connecting" sourceClassName="chip" />
                  <LegacyStatus id="input-chip" initial="Waiting for ROS" sourceClassName="chip muted" />
                </div>
              )}
            >
              <div className="calibration-viewer">
                <canvas id="camera-canvas" />
                <div id="camera-placeholder" className="calibration-placeholder">Waiting for camera image…</div>
              </div>
              <div className="calibration-viewer-meta">
                <span id="frame-meta">No frame</span>
                <span id="coordinate-hint">Freeze a synchronized frame before selecting points.</span>
              </div>
            </Panel>
          )}
          secondary={(
            <Panel bodyLayout="column" className="calibration-controls-panel" fill padding="none" title="Calibration steps">
              <ScrollRegion className="calibration-controls" fill>
                <section className="calibration-control-section">
                  <SectionHeader title="1. Capture" />
                  <div className="calibration-actions">
                    <Button id="freeze-button" className="calibration-action" tone="primary" appearance="solid">Freeze synchronized frame</Button>
                    <Button id="live-button" className="calibration-action">Live</Button>
                  </div>
                </section>

                <section className="calibration-control-section">
                  <SectionHeader title="2. Match markers" />
                  <FormField
                    label="Marker assigned to next click"
                    description="Select a marker, then click its center in the image. Each marker can be used once."
                  >
                    <Select id="marker-select" disabled><option>Freeze a frame first</option></Select>
                  </FormField>
                  <div className="calibration-actions">
                    <Button id="remove-button" className="calibration-action" disabled>Remove last</Button>
                    <Button id="clear-button" className="calibration-action" disabled>Clear</Button>
                  </div>
                  <DataTable className="calibration-table">
                    <table>
                      <thead><tr><th>Marker</th><th>u</th><th>v</th><th>Error</th></tr></thead>
                      <tbody id="points-body"><tr><td colSpan={4} className="empty">No correspondences</td></tr></tbody>
                    </table>
                  </DataTable>
                </section>

                <section className="calibration-control-section">
                  <SectionHeader title="3. Solve" />
                  <Button id="solve-button" className="calibration-wide" tone="primary" appearance="solid" disabled>Solve and save</Button>
                  <LegacyCodeResult id="result-box" initial="Select at least four markers." />
                </section>

                <details className="calibration-details">
                  <summary>ROS and output details</summary>
                  <DescriptionList className="calibration-description-list">
                    <DescriptionItem label="Image" value={<span id="image-topic">—</span>} />
                    <DescriptionItem label="Intrinsic file" value={<span id="intrinsic-file">—</span>} />
                    <DescriptionItem label="Pose prefix" value={<span id="pose-prefix">—</span>} />
                    <DescriptionItem label="Output" value={<span id="output-file">—</span>} />
                  </DescriptionList>
                </details>
              </ScrollRegion>
            </Panel>
          )}
        />
      </div>
    </AppShell>
  )
}

const root = document.getElementById('app')
if (!root) throw new Error('Camera calibration root is unavailable')
flushSync(() => createRoot(root).render(__CALIBRATION_PAGE__ === 'intrinsic' ? <IntrinsicPage /> : <ExtrinsicPage />))
if (__CALIBRATION_PAGE__ === 'intrinsic') void import('./intrinsic-legacy')
else void import('./extrinsic-legacy')
