import assert from "node:assert/strict";
import { mkdtempSync, rmdirSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { it } from "node:test";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import register from "../src/index.ts";
import { startCaptureServer, waitForRequests } from "./helpers.ts";

it("exports configured tags on prompt and idle compaction traces", { timeout: 15000 }, async () => {
  const capture = await startCaptureServer();
  const agentDir = mkdtempSync(join(tmpdir(), "pi-lf-tags-"));
  const vars = [
    "PI_CODING_AGENT_DIR", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL",
    "LANGFUSE_HOST", "LANGFUSE_TRACING_ENABLED", "LANGFUSE_USER_ID", "LANGFUSE_TRACING_ENVIRONMENT",
    "LANGFUSE_RELEASE", "LANGFUSE_PI_TRACEPARENT", "LANGFUSE_PI_PARENT_TRACE_ID",
    "LANGFUSE_PI_PARENT_SPAN_ID", "LANGFUSE_PI_PARENT_SESSION_ID", "LANGFUSE_PI_PARENT_DEPTH",
    "LANGFUSE_PI_PARENT_EXTERNAL_TRACE",
  ];
  const saved = Object.fromEntries(vars.map((key) => [key, process.env[key]]));
  for (const key of vars) delete process.env[key];
  process.env.PI_CODING_AGENT_DIR = agentDir;
  writeFileSync(join(agentDir, "langfuse.json"), JSON.stringify({
    publicKey: "pk-lf-test", secretKey: "sk-lf-test",
    baseUrl: `http://127.0.0.1:${capture.port}`, tags: ["device:laptop"],
  }));
  type Handler = (event: unknown, ctx: ExtensionContext) => unknown;
  const handlers = new Map<string, Handler>();
  const pi = { on: (name: string, handler: Handler) => handlers.set(name, handler) };
  const ctx = {
    hasUI: false,
    cwd: agentDir,
    sessionManager: { getSessionId: () => "tags-session", getEntries: () => [] },
  } as unknown as ExtensionContext;
  try {
    register(pi as unknown as ExtensionAPI);
    await handlers.get("before_agent_start")!({ prompt: "Inspect the project" }, ctx);
    await handlers.get("agent_settled")!({}, ctx);
    await handlers.get("session_compact")!({ compactionEntry: { summary: "Earlier work" } }, ctx);
    await waitForRequests(capture, 2);
    const spans = capture.spans();
    assert.deepEqual(new Set(spans.map((span) => span.name)), new Set(["Conversational Turn", "Compaction"]));
    for (const span of spans) assert.deepEqual(span.attrs["langfuse.trace.tags"], ["pi", "device:laptop"]);
  } finally {
    await handlers.get("session_shutdown")?.({ reason: "quit" }, ctx);
    capture.close();
    for (const key of vars) {
      if (saved[key] === undefined) delete process.env[key];
      else process.env[key] = saved[key];
    }
    unlinkSync(join(agentDir, "langfuse.json"));
    rmdirSync(agentDir);
  }
});
