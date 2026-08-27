# Langfuse Observability Plugin for pi

This extension sends [pi coding agent](https://pi.dev) sessions to
[Langfuse](https://langfuse.com). It records the user prompts, the agent turns,
the model generations with their tokens and cost, and the tool calls.

The full setup guide is the
[Pi Agent integration page](https://langfuse.com/integrations/developer-tools/pi-agent),
which also covers troubleshooting.

> [!WARNING]
> This is an experimental release, so future versions can bring breaking changes.

## What can this integration trace?

The plugin listens to Pi's core lifecycle events and sends every user prompt to
Langfuse as its own trace:

- **Agent turns**: one trace per user prompt, with all turns of a Pi session
  grouped under one session ID. Turn numbering survives a Pi restart.
- **Model generations**: every model request with inputs, outputs, cost, time to
  first token, and token usage including cache-read and reasoning splits.
- **Tool calls**: each tool Pi invokes, with input, output, and an `ERROR` level
  when the call fails.
- **Images**: images you add to a prompt are uploaded as Langfuse media and
  render inside the trace.
- **Subagents**: Pi processes spawned by other extensions nest under the turn
  that started them.

## Prerequisites

Pi needs Node.js 22.0.0 or newer, and pi's package installer shells out to
`npm`, so npm has to be on your `PATH`.

## Install

```bash
pi install npm:@langfuse/pi-observability-plugin
```

To try it for a single run, without installing:

```bash
pi -e npm:@langfuse/pi-observability-plugin
```

## Add your Langfuse credentials

Create a credentials file at `~/.pi/agent/langfuse.json`:

```json
{
  "publicKey": "pk-lf-...",
  "secretKey": "sk-lf-...",
  "baseUrl": "https://cloud.langfuse.com",
  "userId": "your-user-id",
  "environment": "development"
}
```

Only `publicKey` and `secretKey` are required. If `baseUrl` is omitted, the
plugin uses `https://cloud.langfuse.com` (EU region). `userId`, `environment`
and `release` are optional labels that let you segment traces by teammate, stage
or version. Keep the file private, because it holds a secret key.

Alternatively, set your credentials with environment variables:

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_BASE_URL="https://cloud.langfuse.com"
export LANGFUSE_TRACING_ENVIRONMENT="development"
export LANGFUSE_USER_ID="your-user-id"
```

Environment variables take precedence over the config file, so you can override
a single value without editing the file.

## Environment variables

| Variable                       | Description                                                                                                                                       | Required            |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| `LANGFUSE_PUBLIC_KEY`          | Your Langfuse public key (`pk-lf-...`)                                                                                                            | Yes                 |
| `LANGFUSE_SECRET_KEY`          | Your Langfuse secret key (`sk-lf-...`)                                                                                                            | Yes                 |
| `LANGFUSE_BASE_URL`            | Langfuse host. EU: `https://cloud.langfuse.com`, US: `https://us.cloud.langfuse.com`, Japan: `https://jp.cloud.langfuse.com`, HIPAA: `https://hipaa.cloud.langfuse.com` | No (defaults to EU) |
| `LANGFUSE_TRACING_ENVIRONMENT` | Environment label for the traces (e.g. `production`)                                                                                              | No                  |
| `LANGFUSE_USER_ID`             | User ID attached to all traces                                                                                                                    | No                  |

## Enable and disable tracing

| Scope                   | How                                                         |
| ----------------------- | ----------------------------------------------------------- |
| One run                 | `LANGFUSE_TRACING_ENABLED=false pi`                         |
| The current shell       | `export LANGFUSE_TRACING_ENABLED=false` (undo with `unset`) |
| Global or project scope | `pi config`, then switch the extension off                  |
| Remove the plugin       | `pi remove npm:@langfuse/pi-observability-plugin`           |

The kill switch has priority over environment keys and the config file. When
tracing is off, the status line shows `langfuse: off (no keys)` and nothing is
sent. This message reads the same whether the kill switch is set or the keys are
genuinely missing. To remove stored keys, delete `~/.pi/agent/langfuse.json`.

`pi remove` needs the same source you installed from. For a local checkout, see
[Development](#development).

## Development

```bash
pnpm install
pnpm run typecheck
pnpm test
```

The package ships raw TypeScript and pi reads `src/index.ts` directly, so there
is no build step. Therefore `files` in `package.json` holds only `src`,
`README.md` and `LICENSE`.

To run a local checkout against pi:

```bash
pi install /absolute/path/to/pi-observability-plugin
```

Remove it again with `pi remove /absolute/path/to/pi-observability-plugin`.

## Release

1. Update the version in `package.json` and `EXTENSION_VERSION` in
   `src/index.ts`, then merge the version bump into `main`.
2. Tag that commit with the matching `v`-prefixed version and push the tag:

   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

   The tag must exactly match the version in `package.json` or the release
   workflow will fail.
3. The [release workflow](./.github/workflows/release.yml) installs
   dependencies, typechecks the package, stages it on npm with provenance,
   and creates a draft GitHub release with generated release notes.
   Prerelease versions are staged with the `next` npm tag; stable versions
   use `latest`.
4. Review the staged package and approve it to publish, or reject it if
   anything is wrong. This can be done through npm's staged packages UI or
   with the npm CLI:

   ```bash
   npm stage list @langfuse/pi-observability-plugin
   npm stage view <stage-id>
   npm stage approve <stage-id>
   # Or: npm stage reject <stage-id>
   ```

   Approval and rejection require npm two-factor authentication.
5. Publish the draft GitHub release after the staged npm package is approved.

## License

[MIT](./LICENSE)
