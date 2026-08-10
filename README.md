# Langfuse Observability Plugin for pi

Traces [pi coding agent](https://pi.dev) sessions to [Langfuse](https://langfuse.com):
user prompts, agent turns, model generations (tokens, cost, time-to-first-token)
and tool calls — using the same trace naming and nesting as the
[Langfuse Claude Code plugin](https://github.com/langfuse/claude-observability-plugin).

**Status: local prototype (LFE-14802), not yet published.**

## Trace model

One Langfuse trace per user prompt, grouped into a session per pi session:

```
[trace]  "Pi - Turn 3 (a1b2c3d4)"         ← turn number survives pi restarts
└─ span  "Conversational Turn"
   ├─ generation "LLM Call 1"              ← usage incl. cache tokens, cost, TTFT
   ├─ tool       "Tool: bash"              ← input/output, ERROR level on failure
   ├─ generation "LLM Call 2"
   └─ ...
```

Improvements over the Claude Code hook (possible because pi gives us live,
typed events instead of transcript files): `costDetails` from pi's client-side
pricing (works for custom providers), `completionStartTime` (TTFT), error
levels + status messages on failed tools/generations, `environment` support,
and extension self-identification in metadata. Lifecycle correctness follows
the best community extensions: traces finalize on `agent_settled` (not
`agent_end`), provider retries supersede open generations instead of leaking
them, dangling observations are closed as `interrupted`, and flushes are
capped at 3s so pi's exit is never blocked.

## Try it locally

```bash
# 1. deps (once)
cd pi-observability-plugin && npm install

# 2. register with pi (once; from any directory)
pi install /absolute/path/to/pi-observability-plugin
# note: pi runs an npm reconcile in this repo on first startup after
# install/changes — the first pi start can take a minute.

# 3a. configure once (recommended): ~/.pi/agent/langfuse.json, chmod 600
{
  "publicKey": "pk-lf-...",
  "secretKey": "sk-lf-...",
  "baseUrl": "https://cloud.langfuse.com",
  "userId": "you",              // optional
  "environment": "dev",         // optional
  "release": "..."              // optional
}
# then just: cd <any-project> && pi

# 3b. or per shell via env vars (override the file — handy for ad-hoc runs)
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_BASE_URL=https://cloud.langfuse.com   # or LANGFUSE_HOST
pi "Summarize this repository"
```

Config precedence: `LANGFUSE_TRACING_ENABLED=false` (kill switch) > env vars >
`~/.pi/agent/langfuse.json` > defaults. The file accepts literal values only
(no env interpolation, no command execution).

## Enable / disable tracing

| Scope | How |
|---|---|
| One run | `LANGFUSE_TRACING_ENABLED=false pi` |
| Current shell | `export LANGFUSE_TRACING_ENABLED=false` (undo: `unset …`) |
| Persistent (per global/project scope) | `pi config` → toggle the extension (pi's built-in resource manager, analogous to Claude Code's plugin enable/disable) |
| Uninstall | `pi remove <path-to-this-repo>` (or delete `~/.pi/agent/langfuse.json`) |

The kill switch wins over both env keys and the config file. When disabled,
the TUI status shows `langfuse: off (no keys)` and nothing is sent.

Remove with `pi remove /absolute/path/to/pi-observability-plugin`.

Scripting note: in `pi -p` (print mode) close stdin (`pi -p "..." < /dev/null`)
— pi waits for EOF on piped stdin.

`PI_LANGFUSE_DEBUG=true` prints extension breadcrumbs to stderr.

## Development

```bash
npm run typecheck   # strict TS check (no build — pi runs the TS directly)
npm test            # unit + config + end-to-end tests (~5s, fully sandboxed)
```

The end-to-end tests spawn the real pi CLI (devDependency) inside throwaway
temp sandboxes (`PI_CODING_AGENT_DIR`), drive it with a deterministic
in-process mock model, capture the actual OTLP export with a fake Langfuse
ingest on an ephemeral port, and assert the full span tree — names, nesting,
types, usage/cost mapping, error levels, session grouping, turn numbering,
kill switch and config-file resolution. Your `~/.pi` is never touched.

The extension ships as raw TypeScript (`src/index.ts`, loaded by pi via jiti —
no build step). Runtime deps: `@langfuse/tracing` + `@langfuse/otel` (v5 OTEL
SDK, isolated `NodeTracerProvider`, never touches the global provider),
`@opentelemetry/sdk-trace-node`.
