# Langfuse Observability Plugin for pi

This extension sends [pi coding agent](https://pi.dev) sessions to
[Langfuse](https://langfuse.com). It records the user prompts, the agent turns,
the model generations with their tokens and cost, and the tool calls.

## Trace model

The extension makes one Langfuse trace for each user prompt. All traces of one pi
session share a session id.

```
[trace]  "Pi - Turn 3 (a1b2c3d4)"         ← the turn number survives a pi restart
└─ span  "Conversational Turn"
   ├─ generation "LLM Call 1"              ← tokens with the cache split, cost, TTFT
   ├─ tool       "Tool: bash"              ← input, output, ERROR level on a failure
   ├─ generation "LLM Call 2"
   └─ ...
```

You can use this extension together with the other extensions from the pi
package gallery. It listens to the pi core events, so it also records the work of
a tool or a subagent that another extension adds.

## Try it locally

```bash
# 1. install the dependencies (one time)
cd pi-observability-plugin && npm install

# 2. register the extension with pi (one time, from any directory)
pi install /absolute/path/to/pi-observability-plugin

# 3a. give the keys in a file (recommended): ~/.pi/agent/langfuse.json, chmod 600
{
  "publicKey": "pk-lf-...",
  "secretKey": "sk-lf-...",
  "baseUrl": "https://cloud.langfuse.com",
  "userId": "you",              // optional
  "environment": "dev",         // optional
  "release": "..."              // optional
}
# then start pi in any project: cd <any-project> && pi

# 3b. or give the keys in the shell. These values replace the file.
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_BASE_URL=https://cloud.langfuse.com   # or LANGFUSE_HOST
pi "Summarize this repository"
```

The extension reads the configuration in this order: the kill switch
`LANGFUSE_TRACING_ENABLED=false` first, then the environment variables, then
`~/.pi/agent/langfuse.json`, then the defaults. The file holds literal values
only. It cannot hold an environment variable and it cannot run a command.

## Enable and disable the tracing

| Scope | How |
|---|---|
| One run | `LANGFUSE_TRACING_ENABLED=false pi` |
| The current shell | `export LANGFUSE_TRACING_ENABLED=false` (to undo: `unset …`) |
| Global or project scope | `pi config`, then switch the extension off |
| Remove the extension | `pi remove /absolute/path/to/pi-observability-plugin` |

The kill switch has priority over the environment keys and over the file. When
the tracing is off, the status line shows `langfuse: off (no keys)` and the
extension sends nothing.

To remove the keys, delete `~/.pi/agent/langfuse.json`.

For a script, close the standard input in print mode: `pi -p "..." < /dev/null`.
pi waits for the end of the file on a piped standard input.

`PI_LANGFUSE_DEBUG=true` writes the steps of the extension to the standard error.

## Subagent nesting

A pi subagent is a second pi process. The subagent example extension starts one
for each task, and so do the gallery orchestration packages. A child process gets
the environment of its parent.

While a turn is active, the extension puts the ids of that turn in the
environment: `LANGFUSE_PI_PARENT_TRACE_ID`, `LANGFUSE_PI_PARENT_SPAN_ID`,
`LANGFUSE_PI_PARENT_SESSION_ID` and `LANGFUSE_PI_PARENT_DEPTH`. A pi process that
finds these ids puts its trace below the turn that started it. Without them, the
child writes a separate trace.

You can also set these variables yourself to attach a run to a trace from
another tool. The ids must belong to the same Langfuse project that the run
exports to. The built-in flow needs no configuration.
