/**
 * "Open in external terminal": the renderer half of the desktop shell's
 * external-terminal launch.
 *
 * The launch is two steps and the renderer holds neither the binary nor the
 * argv at any point:
 *   1. `POST /api/terminal/external` asks the gateway to approve opening the
 *      picked app in `projectDir`. The gateway is the gate (owner, direct-local
 *      request, integrated terminal enabled, an app picked) and answers an
 *      opaque single-use grant.
 *   2. The grant goes to the desktop shell's `externalTerminalAPI.open`, whose
 *      main process redeems it at the gateway and only then starts the app.
 *
 * Every refusal resolves `{ ok: false }` rather than throwing, so the caller
 * can fall back to the integrated terminal.
 */

/** One allowlisted terminal app installed on this machine. */
export interface ExternalTerminalApp {
  id: string
  label: string
}

interface ExternalTerminalBridge {
  detect: () => Promise<{ apps?: unknown }>
  open: (grant: string) => Promise<{ ok?: boolean; app?: string; error?: string }>
}

function bridge(): ExternalTerminalBridge | undefined {
  const api = (window as { externalTerminalAPI?: Partial<ExternalTerminalBridge> }).externalTerminalAPI
  return typeof api?.detect === 'function' && typeof api?.open === 'function'
    ? (api as ExternalTerminalBridge)
    : undefined
}

/**
 * Whether this window can launch an external terminal at all: true only in
 * the desktop shell, whose preload exposes the bridge. A plain browser tab and
 * the PWA have no bridge, so the row and the picker stay hidden there. Read
 * lazily so a test can stub `window.externalTerminalAPI` per case.
 */
export function canOpenExternalTerminal(): boolean {
  return bridge() !== undefined
}

/**
 * The allowlisted terminal apps installed on this machine, as the desktop
 * shell detects them. Empty when there is no bridge, when the shell refuses
 * (a window whose gateway is remote) and on a malformed answer.
 */
export async function detectExternalTerminals(): Promise<ExternalTerminalApp[]> {
  const api = bridge()
  if (!api) return []
  try {
    const answer = await api.detect()
    const apps = Array.isArray(answer?.apps) ? answer.apps : []
    return apps.filter(
      (a): a is ExternalTerminalApp =>
        !!a && typeof (a as ExternalTerminalApp).id === 'string' && typeof (a as ExternalTerminalApp).label === 'string',
    )
  } catch {
    return []
  }
}

/**
 * Open the picked terminal app in `projectDir`. Resolves `{ ok: true }` once
 * the app has started, else `{ ok: false, error }` where `error` is the
 * gateway's refusal code, the shell's launch error, or `'unavailable'`.
 */
export async function openInExternalTerminal(projectDir: string): Promise<{ ok: boolean; error?: string }> {
  const api = bridge()
  if (!api) return { ok: false, error: 'unavailable' }
  let grant: unknown
  try {
    const res = await fetch('/api/terminal/external', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ cwd: projectDir }),
    })
    const body = (await res.json().catch(() => null)) as { grant?: unknown; code?: unknown } | null
    if (!res.ok) return { ok: false, error: typeof body?.code === 'string' ? body.code : `http_${res.status}` }
    grant = body?.grant
  } catch {
    return { ok: false, error: 'request_failed' }
  }
  if (typeof grant !== 'string' || !grant) return { ok: false, error: 'no_grant' }
  try {
    const result = await api.open(grant)
    return result?.ok === true ? { ok: true } : { ok: false, error: result?.error || 'launch_failed' }
  } catch {
    return { ok: false, error: 'launch_failed' }
  }
}
