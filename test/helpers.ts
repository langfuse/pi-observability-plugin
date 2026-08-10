/**
 * Shared test infrastructure — fully self-contained (no external services):
 *
 * - startMockProvider(): OpenAI-compatible chat/completions server with a
 *   deterministic 3-stage script (bash ls → read README.md → final answer,
 *   realistic usage incl. cache tokens). Prompts containing "[fail]" run a
 *   failing command first, so error paths can be asserted.
 * - startCaptureServer(): fake Langfuse ingest that records every request.
 * - runPi(): spawns the real pi CLI (devDependency) in a throwaway sandbox
 *   (PI_CODING_AGENT_DIR = temp dir), loading the extension via `-e`.
 *   The user's ~/.pi is never touched.
 */
import { spawn } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import http from "node:http";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import zlib from "node:zlib";

export const REPO_ROOT = resolve(import.meta.dirname, "..");
const PI_BIN = join(REPO_ROOT, "node_modules", ".bin", "pi");
const EXTENSION = join(REPO_ROOT, "src", "index.ts");

// --------------------------------------------------------------------------
// Mock OpenAI-compatible provider
// --------------------------------------------------------------------------

interface OpenAiMessage {
  role: string;
  content?: unknown;
}

function writeSseEvent(res: http.ServerResponse, obj: unknown) {
  res.write(`data: ${JSON.stringify(obj)}\n\n`);
}

function streamChunks(
  res: http.ServerResponse,
  model: string,
  parts: { text?: string; tool?: { name: string; args: unknown }; finish: string; usage: unknown },
) {
  const base = { id: "chatcmpl-mock", object: "chat.completion.chunk", created: 1, model };
  writeSseEvent(res, { ...base, choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }] });
  if (parts.text) {
    writeSseEvent(res, { ...base, choices: [{ index: 0, delta: { content: parts.text }, finish_reason: null }] });
  }
  if (parts.tool) {
    writeSseEvent(res, {
      ...base,
      choices: [
        {
          index: 0,
          delta: {
            tool_calls: [
              {
                index: 0,
                id: `call_${Math.random().toString(36).slice(2, 10)}`,
                type: "function",
                function: { name: parts.tool.name, arguments: JSON.stringify(parts.tool.args) },
              },
            ],
          },
          finish_reason: null,
        },
      ],
    });
  }
  writeSseEvent(res, { ...base, choices: [{ index: 0, delta: {}, finish_reason: parts.finish }] });
  writeSseEvent(res, { ...base, choices: [], usage: parts.usage });
  res.write("data: [DONE]\n\n");
  res.end();
}

export function startMockProvider(): Promise<{ port: number; close: () => void }> {
  const server = http.createServer((req, res) => {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      const payload = JSON.parse(body) as { model?: string; messages?: OpenAiMessage[] };
      const messages = payload.messages ?? [];
      const model = payload.model ?? "mock-gpt-1";
      const lastUserIdx = messages.map((m) => m.role).lastIndexOf("user");
      const stage = messages.slice(lastUserIdx + 1).filter((m) => m.role === "tool").length;
      const lastUser = messages[lastUserIdx];
      const failMode = JSON.stringify(lastUser?.content ?? "").includes("[fail]");

      res.writeHead(200, { "content-type": "text/event-stream" });
      const usage = (p: number, c: number, cached: number) => ({
        prompt_tokens: p,
        completion_tokens: c,
        total_tokens: p + c,
        prompt_tokens_details: { cached_tokens: cached },
      });

      if (failMode && stage === 0) {
        streamChunks(res, model, {
          text: "Reading the log file. ",
          tool: { name: "bash", args: { command: "cat does-not-exist.log" } },
          finish: "tool_calls",
          usage: usage(900, 28, 0),
        });
      } else if (failMode) {
        streamChunks(res, model, {
          text: "The file does not exist, so I stopped.",
          finish: "stop",
          usage: usage(980, 30, 512),
        });
      } else if (stage === 0) {
        streamChunks(res, model, {
          text: "Looking at the project. ",
          tool: { name: "bash", args: { command: "ls" } },
          finish: "tool_calls",
          usage: usage(1200, 32, 0),
        });
      } else if (stage === 1) {
        streamChunks(res, model, {
          text: "Reading the README. ",
          tool: { name: "read", args: { path: "README.md" } },
          finish: "tool_calls",
          usage: usage(1350, 41, 1024),
        });
      } else {
        streamChunks(res, model, {
          text: "This is the test workspace. Done.",
          finish: "stop",
          usage: usage(1600, 78, 1280),
        });
      }
    });
  });
  return new Promise((resolvePromise) => {
    server.listen(0, "127.0.0.1", () => {
      const port = (server.address() as { port: number }).port;
      resolvePromise({ port, close: () => server.close() });
    });
  });
}

// --------------------------------------------------------------------------
// Capture server (fake Langfuse OTLP ingest)
// --------------------------------------------------------------------------

export interface CapturedSpan {
  traceId: string;
  spanId: string;
  parentSpanId?: string;
  name: string;
  attrs: Record<string, unknown>;
  startNs: bigint;
  endNs: bigint;
}

export interface Capture {
  port: number;
  requests: unknown[];
  spans: () => CapturedSpan[];
  close: () => void;
}

function parseAttrValue(v: {
  stringValue?: string;
  intValue?: number | string;
  doubleValue?: number;
  boolValue?: boolean;
  arrayValue?: { values?: Array<never> };
}): unknown {
  if (v.arrayValue) return (v.arrayValue.values ?? []).map((inner) => parseAttrValue(inner));
  return v.stringValue ?? v.intValue ?? v.doubleValue ?? v.boolValue;
}

export function startCaptureServer(): Promise<Capture> {
  const requests: unknown[] = [];
  const server = http.createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => {
      let raw = Buffer.concat(chunks);
      if ((req.headers["content-encoding"] ?? "") === "gzip") raw = zlib.gunzipSync(raw);
      try {
        requests.push(JSON.parse(raw.toString("utf8")));
      } catch {
        requests.push({ _raw: raw.toString("base64") });
      }
      res.writeHead(200, { "content-type": "application/json" });
      res.end("{}");
    });
  });
  return new Promise((resolvePromise) => {
    server.listen(0, "127.0.0.1", () => {
      const port = (server.address() as { port: number }).port;
      resolvePromise({
        port,
        requests,
        spans: () => {
          const out: CapturedSpan[] = [];
          for (const r of requests as Array<{ resourceSpans?: Array<{ scopeSpans?: Array<{ spans?: Array<Record<string, unknown>> }> }> }>) {
            for (const rs of r.resourceSpans ?? []) {
              for (const ss of rs.scopeSpans ?? []) {
                for (const sp of ss.spans ?? []) {
                  const attrs: Record<string, unknown> = {};
                  for (const a of (sp.attributes as Array<{ key: string; value: never }>) ?? []) {
                    attrs[a.key] = parseAttrValue(a.value);
                  }
                  out.push({
                    traceId: sp.traceId as string,
                    spanId: sp.spanId as string,
                    parentSpanId: (sp.parentSpanId as string) || undefined,
                    name: sp.name as string,
                    attrs,
                    startNs: BigInt((sp.startTimeUnixNano as string) ?? 0),
                    endNs: BigInt((sp.endTimeUnixNano as string) ?? 0),
                  });
                }
              }
            }
          }
          return out;
        },
        close: () => server.close(),
      });
    });
  });
}

// --------------------------------------------------------------------------
// pi sandbox runner
// --------------------------------------------------------------------------

export interface Sandbox {
  agentDir: string;
  workspace: string;
}

export function createSandbox(mockPort: number): Sandbox {
  const agentDir = mkdtempSync(join(tmpdir(), "pi-lf-agent-"));
  const workspace = mkdtempSync(join(tmpdir(), "pi-lf-ws-"));
  writeFileSync(join(workspace, "README.md"), "# Test workspace\nUsed by integration tests.\n");
  writeFileSync(
    join(agentDir, "models.json"),
    JSON.stringify({
      providers: {
        mock: {
          name: "Mock",
          baseUrl: `http://127.0.0.1:${mockPort}/v1`,
          api: "openai-completions",
          apiKey: "mock-key",
          models: [
            {
              id: "mock-gpt-1",
              name: "Mock GPT 1",
              reasoning: false,
              input: ["text"],
              contextWindow: 128000,
              maxTokens: 8192,
              cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
            },
          ],
        },
      },
    }),
  );
  return { agentDir, workspace };
}

/**
 * Async on purpose: the mock provider and capture server run inside the test
 * process — a synchronous spawn would freeze the event loop and deadlock pi
 * against its own model backend.
 */
export function runPi(
  sandbox: Sandbox,
  prompt: string,
  opts: { continue?: boolean; env?: Record<string, string | undefined> } = {},
): Promise<{ status: number | null; stdout: string; stderr: string }> {
  const args = [
    "-e",
    EXTENSION,
    "--provider",
    "mock",
    "--model",
    "mock-gpt-1",
    "--api-key",
    "mock-key",
    "--no-context-files",
  ];
  if (opts.continue) args.push("-c");
  args.push("-p", prompt);

  return new Promise((resolvePromise) => {
    const child = spawn(PI_BIN, args, {
      cwd: sandbox.workspace,
      // stdin must be closed — pi's print mode waits for EOF on piped stdin.
      stdio: ["ignore", "pipe", "pipe"],
      env: {
        ...process.env,
        PI_CODING_AGENT_DIR: sandbox.agentDir,
        PI_SKIP_VERSION_CHECK: "1",
        PI_TELEMETRY: "0",
        LANGFUSE_PUBLIC_KEY: undefined,
        LANGFUSE_SECRET_KEY: undefined,
        LANGFUSE_BASE_URL: undefined,
        LANGFUSE_HOST: undefined,
        LANGFUSE_USER_ID: undefined,
        LANGFUSE_TRACING_ENVIRONMENT: undefined,
        LANGFUSE_RELEASE: undefined,
        LANGFUSE_TRACING_ENABLED: undefined,
        ...opts.env,
      } as NodeJS.ProcessEnv,
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (d: Buffer) => (stdout += d.toString()));
    child.stderr.on("data", (d: Buffer) => (stderr += d.toString()));
    const killer = setTimeout(() => child.kill("SIGKILL"), 60_000);
    child.on("close", (code) => {
      clearTimeout(killer);
      resolvePromise({ status: code, stdout, stderr });
    });
  });
}

/** Poll until the capture holds at least `n` export requests (flushes are async). */
export async function waitForRequests(capture: Capture, n: number, timeoutMs = 5000): Promise<void> {
  const start = Date.now();
  while (capture.requests.length < n && Date.now() - start < timeoutMs) {
    await new Promise((r) => setTimeout(r, 100));
  }
}
