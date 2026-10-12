/**
 * The Files tab's "Open in external terminal" row.
 *
 * The row is an opt-in beside the integrated terminal, never in its place:
 * with no app picked the Project actions menu is exactly what it was, and the
 * row appears only in the desktop shell, against a gateway on this machine,
 * once an app is picked. A refused or failed launch falls back to the
 * integrated terminal.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, within, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const H = vi.hoisted(() => ({
  config: {} as Record<string, unknown>,
  api: {
    projectTree: vi.fn(),
    projectGitStatus: vi.fn(),
    revealPath: vi.fn(),
    kirocrewConfig: vi.fn(),
  },
}))

vi.mock('../api/client', () => ({ api: H.api, ApiError: class ApiError extends Error {} }))

const brandingEnv = vi.hoisted(() => ({ directLocal: true }))
vi.mock('../hooks/useBranding', () => ({
  useBranding: () => ({ botName: 'Test', avatar: '', directLocal: brandingEnv.directLocal }),
}))

vi.mock('../pierre/tree', () => ({
  TreeSkeleton: () => null,
  PierreWorkspaceTree: () => <div data-testid="tree" />,
}))

import FilesHomePanel from '../pages/chat/FilesHomePanel'

const DIR = '/repo/my-project'
const GRANT = 'g'.repeat(43)
const TERMINAL_ROW = 'Open terminal in the project directory'
const EXTERNAL_ROW = 'Open in kitty'

const header = () => screen.getByText('Files').parentElement as HTMLElement

type Bridge = { detect: ReturnType<typeof vi.fn>; open: ReturnType<typeof vi.fn> }
let bridge: Bridge | undefined
let fetchMock: ReturnType<typeof vi.fn>

function installBridge(openResult: unknown = { ok: true, app: 'kitty' }) {
  bridge = {
    detect: vi.fn().mockResolvedValue({ apps: [{ id: 'kitty', label: 'kitty' }] }),
    open: vi.fn().mockResolvedValue(openResult),
  }
  ;(window as { externalTerminalAPI?: Bridge }).externalTerminalAPI = bridge
}

function pick(app: string | undefined) {
  H.config = app === undefined ? {} : { dashboard: { terminal: { external_app: app } } }
}

/** `null` mounts the panel the way a host with no integrated terminal does. */
function mount(onOpenTerminal: (() => void) | null = vi.fn()) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={qc}>
      <FilesHomePanel projectDir={DIR} onFileOpen={vi.fn()} onOpenTerminal={onOpenTerminal ?? undefined} />
    </QueryClientProvider>,
  )
  return { onOpenTerminal }
}

function openProjectActions() {
  fireEvent.keyDown(within(header()).getByLabelText('Project actions'), { key: 'Enter' })
}

/** The menu's rows by their first line, in order, once it has opened. */
async function menuRows() {
  await screen.findByText(TERMINAL_ROW)
  return screen.getAllByRole('menuitem').map(el => el.textContent ?? '')
}

beforeEach(() => {
  H.api.projectTree.mockReset().mockResolvedValue({ root: DIR, paths: [], repo: true })
  H.api.projectGitStatus.mockReset().mockResolvedValue({ repo: true, files: [] })
  H.api.revealPath.mockReset().mockResolvedValue(undefined)
  H.api.kirocrewConfig.mockReset().mockImplementation(async () => H.config)
  brandingEnv.directLocal = true
  pick(undefined)
  bridge = undefined
  delete (window as { externalTerminalAPI?: unknown }).externalTerminalAPI
  fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({ grant: GRANT, app: 'kitty' }) })
  vi.stubGlobal('fetch', fetchMock)
})

afterEach(() => {
  vi.unstubAllGlobals()
  delete (window as { externalTerminalAPI?: unknown }).externalTerminalAPI
})

describe('with no external app picked (the default)', () => {
  it('leaves the Project actions menu and its terminal row exactly as they were', async () => {
    installBridge()
    pick(undefined)
    const { onOpenTerminal } = mount()
    await waitFor(() => expect(H.api.kirocrewConfig).toHaveBeenCalled())
    openProjectActions()
    const rows = await menuRows()
    expect(rows).toHaveLength(1)
    expect(rows[0]).toContain(TERMINAL_ROW)
    expect(screen.queryByText(EXTERNAL_ROW)).toBeNull()
    fireEvent.click(screen.getByText(TERMINAL_ROW))
    expect(onOpenTerminal).toHaveBeenCalledTimes(1)
    expect(fetchMock).not.toHaveBeenCalled()
    expect(bridge!.open).not.toHaveBeenCalled()
  })

  it('treats an empty pick as unset', async () => {
    installBridge()
    pick('')
    mount()
    await waitFor(() => expect(H.api.kirocrewConfig).toHaveBeenCalled())
    openProjectActions()
    expect(await menuRows()).toHaveLength(1)
  })
})

describe('with an app picked', () => {
  it('adds the row below the integrated terminal row on the desktop app', async () => {
    installBridge()
    pick('kitty')
    mount()
    openProjectActions()
    await screen.findByText(EXTERNAL_ROW)
    const rows = screen.getAllByRole('menuitem').map(el => el.textContent ?? '')
    expect(rows).toHaveLength(2)
    expect(rows[0]).toContain(TERMINAL_ROW)
    expect(rows[1]).toContain(EXTERNAL_ROW)
  })

  it('names the picked app by the label the desktop shell detected', async () => {
    installBridge()
    bridge!.detect.mockResolvedValue({ apps: [{ id: 'wezterm', label: 'WezTerm' }] })
    pick('wezterm')
    mount()
    openProjectActions()
    expect(await screen.findByText('Open in WezTerm')).toBeInTheDocument()
    expect(screen.getByText('Opens in its own window, in this project’s folder.')).toBeInTheDocument()
  })

  it('shows no row for a pick that is no longer installed', async () => {
    installBridge()
    bridge!.detect.mockResolvedValue({ apps: [{ id: 'wezterm', label: 'WezTerm' }] })
    pick('kitty')
    mount()
    await waitFor(() => expect(bridge!.detect).toHaveBeenCalled())
    openProjectActions()
    expect(await menuRows()).toHaveLength(1)
    expect(screen.queryByText(/^Open in /)).toBeNull()
  })

  it('is absent in the browser build, where there is no launch bridge', async () => {
    pick('kitty')
    mount()
    openProjectActions()
    expect(await menuRows()).toHaveLength(1)
    expect(screen.queryByText(EXTERNAL_ROW)).toBeNull()
    expect(H.api.kirocrewConfig).not.toHaveBeenCalled()
  })

  it('is absent for a gateway that is not on this machine', async () => {
    installBridge()
    pick('kitty')
    brandingEnv.directLocal = false
    mount()
    openProjectActions()
    expect(await menuRows()).toHaveLength(1)
    expect(screen.queryByText(EXTERNAL_ROW)).toBeNull()
  })

  it('is absent when the host serves no integrated terminal', async () => {
    installBridge()
    pick('kitty')
    mount(null)
    await waitFor(() => expect(screen.getByTestId('tree')).toBeInTheDocument())
    expect(within(header()).queryByLabelText('Project actions')).toBeNull()
  })

  it('asks the gateway for a grant and hands only the grant to the desktop shell', async () => {
    installBridge()
    pick('kitty')
    const { onOpenTerminal } = mount()
    openProjectActions()
    fireEvent.click(await screen.findByText(EXTERNAL_ROW))
    await waitFor(() => expect(bridge!.open).toHaveBeenCalledWith(GRANT))
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(url).toBe('/api/terminal/external')
    expect(init.method).toBe('POST')
    expect(JSON.parse(String(init.body))).toEqual({ cwd: DIR })
    expect(onOpenTerminal).not.toHaveBeenCalled()
    expect(screen.queryByTestId('files-home-external-terminal-fallback')).toBeNull()
  })

  it('falls back to the integrated terminal when the gateway refuses', async () => {
    installBridge()
    pick('kitty')
    fetchMock.mockResolvedValue({
      ok: false,
      status: 403,
      json: async () => ({ code: 'external_terminal_not_local' }),
    })
    const { onOpenTerminal } = mount()
    openProjectActions()
    fireEvent.click(await screen.findByText(EXTERNAL_ROW))
    await waitFor(() => expect(onOpenTerminal).toHaveBeenCalledTimes(1))
    expect(bridge!.open).not.toHaveBeenCalled()
    expect(await screen.findByTestId('files-home-external-terminal-fallback')).toHaveTextContent(
      "Couldn't open kitty, so the built-in terminal opened instead.",
    )
  })

  it('falls back to the integrated terminal when the launch fails', async () => {
    installBridge({ ok: false, error: 'launch-failed' })
    pick('kitty')
    const { onOpenTerminal } = mount()
    openProjectActions()
    fireEvent.click(await screen.findByText(EXTERNAL_ROW))
    await waitFor(() => expect(onOpenTerminal).toHaveBeenCalledTimes(1))
    expect(bridge!.open).toHaveBeenCalledWith(GRANT)
    expect(await screen.findByTestId('files-home-external-terminal-fallback')).toBeInTheDocument()
  })

  it('falls back when the desktop shell rejects the request', async () => {
    installBridge()
    bridge!.open.mockRejectedValue(new Error('external-terminal:open is restricted to the local dashboard'))
    pick('kitty')
    const { onOpenTerminal } = mount()
    openProjectActions()
    fireEvent.click(await screen.findByText(EXTERNAL_ROW))
    await waitFor(() => expect(onOpenTerminal).toHaveBeenCalledTimes(1))
  })
})
