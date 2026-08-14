/**
 * End-to-end tests: real pi CLI (devDependency) + deterministic mock model
 * + fake Langfuse ingest, all sandboxed in temp dirs. Asserts the actual
 * exported OTLP span tree — names, nesting, types, usage, cost, errors,
 * session grouping and turn numbering.
 */
import assert from "node:assert/strict";
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { after, before, describe, it } from "node:test";
import {
  type Capture,
  type CapturedSpan,
  createSandbox,
  runPi,
  startCaptureServer,
  startMockProvider,
  waitForRequests,
} from "./helpers.ts";

function buildLangfuseEnv(capture: Capture): Record<string, string> {
  return {
    LANGFUSE_PUBLIC_KEY: "pk-lf-test",
    LANGFUSE_SECRET_KEY: "sk-lf-test",
    LANGFUSE_BASE_URL: `http://127.0.0.1:${capture.port}`,
    LANGFUSE_USER_ID: "test-user",
  };
}

function findSpansByName(spans: CapturedSpan[], name: string): CapturedSpan[] {
  return spans.filter((s) => s.name === name);
}

describe("integration: pi -> extension -> Langfuse export", () => {
  let mock: { port: number; close: () => void };

  before(async () => {
    mock = await startMockProvider();
  });
  after(() => mock.close());

  it("exports the Claude-hook-shaped trace tree for a happy-path prompt", async () => {
    const capture = await startCaptureServer();
    try {
      const sandbox = createSandbox(mock.port);
      const result = await runPi(sandbox, "Explore this project and summarize it", { env: buildLangfuseEnv(capture) });
      assert.equal(result.status, 0, `pi failed: ${result.stderr}`);
      await waitForRequests(capture, 1);
      const spans = capture.spans();

      // Root
      const roots = findSpansByName(spans, "Conversational Turn");
      assert.equal(roots.length, 1);
      const root = roots[0]!;
      assert.equal(root.parentSpanId, undefined);
      assert.equal(root.attrs["langfuse.observation.type"], "span");
      assert.match(String(root.attrs["langfuse.trace.name"]), /^Pi - Turn 1 \(.+\)$/);
      assert.equal(root.attrs["user.id"], "test-user");
      assert.ok(root.attrs["session.id"], "sessionId must be set");
      assert.deepEqual(root.attrs["langfuse.trace.tags"], ["pi"]);

      // Self identification
      const metadata = (k: string) => root.attrs[`langfuse.observation.metadata.${k}`];
      assert.equal(metadata("source"), "pi");
      assert.equal(metadata("extension"), "@langfuse/pi-observability-plugin");
      // OTLP JSON serializes numbers/booleans in metadata as strings.
      assert.equal(Number(metadata("turn_number")), 1);

      // Children: 3 generations + 2 tools, all direct children of the root (flat)
      const tree = spans.filter((s) => s.traceId === root.traceId);
      const generations = findSpansByName(tree, "LLM Call");
      const tools = tree.filter((s) => s.name.startsWith("Tool: "));
      assert.equal(generations.length, 3);
      assert.deepEqual(tools.map((t) => t.name).sort(), ["Tool: bash", "Tool: read"]);
      for (const child of [...generations, ...tools]) {
        assert.equal(child.parentSpanId, root.spanId, `${child.name} must be a direct child of the root`);
      }
      for (const generation of generations) {
        assert.equal(generation.attrs["langfuse.observation.type"], "generation");
        assert.equal(generation.attrs["langfuse.observation.model.name"], "mock-gpt-1");
        assert.ok(generation.attrs["langfuse.observation.completion_start_time"], "TTFT must be set");
      }
      for (const tool of tools) {
        assert.equal(tool.attrs["langfuse.observation.type"], "tool");
      }

      // Usage/cost mapping (second generation has cache reads)
      const usages = generations.map(
        (g) => JSON.parse(String(g.attrs["langfuse.observation.usage_details"])) as Record<string, number>,
      );
      const cached = usages.find((u) => u.cache_read_input_tokens === 1024);
      assert.ok(cached, "cache_read_input_tokens must be mapped");
      assert.equal(cached?.input, 326, "input must exclude cached tokens (no double counting)");
      // The mock gives 30 reasoning tokens to the last generation. This makes
      // sure that the split is correct through the real pi pipeline.
      const reasoned = usages.find((u) => u.output_reasoning_tokens !== undefined);
      assert.deepEqual(
        reasoned,
        { input: 320, output: 48, output_reasoning_tokens: 30, cache_read_input_tokens: 1280 },
        "reasoning tokens must be split out of output end-to-end",
      );
      const costs = generations.map(
        (g) => JSON.parse(String(g.attrs["langfuse.observation.cost_details"])) as Record<string, number>,
      );
      assert.ok(
        costs.some((c) => Math.abs((c.total ?? 0) - 0.0019002) < 1e-9),
        "buildCostDetails must carry pi's client-side pricing",
      );
      const cachedCost = costs.find((c) => c.cache_read_input_tokens !== undefined);
      assert.ok(cachedCost, "cache cost must use the canonical usage key spelling");
      assert.ok(
        Math.abs((cachedCost?.cache_read_input_tokens ?? 0) - 0.0003072) < 1e-9,
        "cache_read_input_tokens cost must match the mock price table",
      );

      // Every span is closed with a real duration
      for (const span of tree) {
        assert.ok(span.endNs > span.startNs, `${span.name} must have a real duration`);
      }
    } finally {
      capture.close();
    }
  });

  it("continues the session: turn 2 shares the session id and increments the turn number", async () => {
    const capture = await startCaptureServer();
    try {
      const sandbox = createSandbox(mock.port);
      const env = buildLangfuseEnv(capture);
      assert.equal((await runPi(sandbox, "First prompt", { env })).status, 0);
      assert.equal((await runPi(sandbox, "Second prompt", { env, continue: true })).status, 0);
      await waitForRequests(capture, 2);
      const roots = findSpansByName(capture.spans(), "Conversational Turn");

      assert.equal(roots.length, 2);
      const [t1, t2] = [...roots].sort((a, b) => (a.startNs < b.startNs ? -1 : 1));
      assert.notEqual(t1!.traceId, t2!.traceId, "each prompt is its own trace");
      assert.equal(t1!.attrs["session.id"], t2!.attrs["session.id"], "same pi session");
      assert.match(String(t1!.attrs["langfuse.trace.name"]), /Turn 1/);
      assert.match(String(t2!.attrs["langfuse.trace.name"]), /Turn 2/, "turn number must survive pi -c restarts");
    } finally {
      capture.close();
    }
  });

  it("marks failed tools and the root with ERROR level", async () => {
    const capture = await startCaptureServer();
    try {
      const sandbox = createSandbox(mock.port);
      assert.equal((await runPi(sandbox, "[fail] read the log file", { env: buildLangfuseEnv(capture) })).status, 0);
      await waitForRequests(capture, 1);
      const spans = capture.spans();

      const tool = findSpansByName(spans, "Tool: bash")[0];
      assert.ok(tool, "failing bash tool must be traced");
      assert.equal(tool!.attrs["langfuse.observation.level"], "ERROR");
      assert.equal(tool!.attrs["langfuse.observation.status_message"], "Tool execution failed");
      assert.equal(String(tool!.attrs["langfuse.observation.metadata.is_error"]), "true");

      const root = findSpansByName(spans, "Conversational Turn")[0];
      assert.equal(root!.attrs["langfuse.observation.level"], "ERROR", "root reflects that the turn saw an error");
    } finally {
      capture.close();
    }
  });

  it("kill switch: LANGFUSE_TRACING_ENABLED=false exports nothing", async () => {
    const capture = await startCaptureServer();
    try {
      const sandbox = createSandbox(mock.port);
      const result = await runPi(sandbox, "This run must not be traced", {
        env: { ...buildLangfuseEnv(capture), LANGFUSE_TRACING_ENABLED: "false" },
      });
      assert.equal(result.status, 0);
      await new Promise((r) => setTimeout(r, 1500));
      assert.equal(capture.requests.length, 0, "no exports with the kill switch on");
    } finally {
      capture.close();
    }
  });

  it("reads credentials from the agent-dir config file when no env vars are set", async () => {
    const capture = await startCaptureServer();
    try {
      const sandbox = createSandbox(mock.port);
      writeFileSync(
        join(sandbox.agentDir, "langfuse.json"),
        JSON.stringify({
          publicKey: "pk-lf-file",
          secretKey: "sk-lf-file",
          baseUrl: `http://127.0.0.1:${capture.port}`,
          userId: "file-user",
        }),
      );
      assert.equal((await runPi(sandbox, "Trace me via config file")).status, 0);
      await waitForRequests(capture, 1);
      const root = findSpansByName(capture.spans(), "Conversational Turn")[0];
      assert.ok(root, "config-file-only run must export a trace");
      assert.equal(root!.attrs["user.id"], "file-user");
    } finally {
      capture.close();
    }
  });
});
