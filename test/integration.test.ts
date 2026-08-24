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
  REPO_ROOT,
  createSandbox,
  SUMMARIZATION_USAGE,
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
      assert.equal(root.attrs["langfuse.trace.name"], "Pi Turn");
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
      assert.equal(Number(t1!.attrs["langfuse.observation.metadata.turn_number"]), 1);
      assert.equal(
        Number(t2!.attrs["langfuse.observation.metadata.turn_number"]),
        2,
        "turn number must survive pi -c restarts",
      );
      assert.equal(
        t1!.attrs["langfuse.trace.name"],
        t2!.attrs["langfuse.trace.name"],
        "trace name is constant so Langfuse name-based grouping works",
      );
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

  it("nests a spawned subagent into the parent trace instead of orphaning it", async () => {
    const capture = await startCaptureServer();
    try {
      const sandbox = createSandbox(mock.port);
      const result = await runPi(sandbox, "Delegate the repo inspection, then summarize", {
        env: buildLangfuseEnv(capture),
        extensions: [join(REPO_ROOT, "test", "fixtures", "subagent-tool.ts")],
      });
      assert.equal(result.status, 0, `pi failed: ${result.stderr}`);
      // The parent and the child export separately. Wait for the two exports.
      await waitForRequests(capture, 2, 15_000);
      const spans = capture.spans();

      const parentRoot = findSpansByName(spans, "Conversational Turn")[0];
      const subagentRoot = findSpansByName(spans, "Subagent Turn")[0];
      assert.ok(parentRoot, "parent turn must be traced");
      assert.ok(subagentRoot, "subagent turn must be traced");

      // The child must join the parent trace.
      assert.equal(subagentRoot!.traceId, parentRoot!.traceId, "subagent must share the parent's trace id");
      assert.equal(
        subagentRoot!.parentSpanId,
        parentRoot!.spanId,
        "subagent root must be a child of the parent turn root",
      );
      assert.equal(parentRoot!.parentSpanId, undefined, "the parent turn stays the trace root");

      // Only the real root sets the trace fields.
      assert.ok(parentRoot!.attrs["langfuse.trace.name"], "parent owns the trace name");
      assert.equal(
        subagentRoot!.attrs["langfuse.trace.name"],
        undefined,
        "subagent must not overwrite the trace name it is nested into",
      );
      assert.equal(subagentRoot!.attrs["session.id"], undefined, "subagent must not set the session");
      assert.equal(subagentRoot!.attrs["user.id"], undefined, "subagent must not set the user");
      assert.equal(subagentRoot!.attrs["langfuse.trace.tags"], undefined, "subagent must not set tags");
      const meta = (k: string) => subagentRoot!.attrs[`langfuse.observation.metadata.${k}`];
      assert.equal(String(meta("pi_subagent")), "true");
      assert.equal(Number(meta("subagent_depth")), 1);

      // The model calls of the subagent go in the same trace. Their tokens add
      // to the parent totals.
      const subagentChildren = spans.filter((s) => s.parentSpanId === subagentRoot!.spanId);
      assert.ok(
        subagentChildren.some((s) => s.attrs["langfuse.observation.type"] === "generation"),
        "the subagent's generations must be nested under it",
      );
      for (const span of subagentChildren) {
        assert.equal(span.traceId, parentRoot!.traceId, `${span.name} must stay in the parent trace`);
      }

      assert.ok(findSpansByName(spans, "Tool: subagent")[0], "the delegating tool call must be traced");
    } finally {
      capture.close();
    }
  });

  it("publishes the parent context only while the turn is active", async () => {
    const capture = await startCaptureServer();
    try {
      const sandbox = createSandbox(mock.port);
      const result = await runPi(sandbox, "Explore this project and summarize it", {
        env: buildLangfuseEnv(capture),
        extensions: [join(REPO_ROOT, "test", "fixtures", "env-probe.ts")],
      });
      assert.equal(result.status, 0, `pi failed: ${result.stderr}`);
      // The probe runs after our extension, so the withdraw has already happened
      // when it prints at agent_settled.
      assert.match(result.stderr, /PROBE turn [0-9a-f]{32}/, "parent ids must be published during the turn");
      assert.match(result.stderr, /PROBE settled <unset>/, "parent ids must be withdrawn after the turn");
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

  // Compaction calls the provider through completeSummarization, bypassing the
  // agent loop, so no message_end fires — yet pi still books the tokens.
  it("traces the compaction summarization call so the session total matches pi", async () => {
    const capture = await startCaptureServer();
    try {
      const sandbox = createSandbox(mock.port, {
        contextWindow: 18000, // minus reserveTokens 16384 -> threshold 1616
        keepRecentTokens: 300,
        readmeFillerLines: 200,
      });
      const result = await runPi(sandbox, "Explore this project and summarize it", {
        env: buildLangfuseEnv(capture),
      });
      assert.equal(result.status, 0, `pi failed: ${result.stderr}`);
      await waitForRequests(capture, 1);
      const spans = capture.spans();

      const compactions = findSpansByName(spans, "Compaction");
      assert.equal(compactions.length, 1, "the compaction summarization call must be traced");
      const compaction = compactions[0]!;
      assert.equal(compaction.attrs["langfuse.observation.type"], "generation");

      const usage = JSON.parse(String(compaction.attrs["langfuse.observation.usage_details"]));
      assert.deepEqual(usage, {
        input: SUMMARIZATION_USAGE.prompt,
        output: SUMMARIZATION_USAGE.completion,
      });
      const cost = JSON.parse(String(compaction.attrs["langfuse.observation.cost_details"]));
      // 3571 input at $3/M + 313 output at $15/M, the sandbox rate card.
      const expected = (3 / 1e6) * SUMMARIZATION_USAGE.prompt + (15 / 1e6) * SUMMARIZATION_USAGE.completion;
      assert.ok(Math.abs(cost.total - expected) < 1e-12, `${cost.total} != ${expected}`);

      // A dropped startTime would collapse the span to ~0.
      assert.ok(compaction.endNs > compaction.startNs, "compaction span must have a duration");

      // Nested under the turn root when the compaction happens inside a turn.
      const root = findSpansByName(spans, "Conversational Turn")[0]!;
      assert.equal(compaction.parentSpanId, root.spanId);
      assert.equal(compaction.traceId, root.traceId);
      assert.equal(compaction.attrs["langfuse.observation.metadata.compaction_reason"], "threshold");
    } finally {
      capture.close();
    }
  });
});
