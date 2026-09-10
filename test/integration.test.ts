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
  type MockProvider,
  type OpenAiMessage,
  REPO_ROOT,
  createSandbox,
  FINAL_ANSWER_THINKING,
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

function byStart(spans: CapturedSpan[]): CapturedSpan[] {
  return [...spans].sort((a, b) => (a.startNs < b.startNs ? -1 : 1));
}

interface TracedMessage {
  role: string;
  content?: string;
  thinking?: Array<{ type: string; content: string; redacted?: true }>;
  tool_calls?: Array<{ id: string; type: string; function: { name: string; arguments?: string } }>;
  tool_call_id?: string;
  name?: string;
}

function inputOf(span: CapturedSpan): TracedMessage[] {
  const raw = span.attrs["langfuse.observation.input"];
  const parsed: unknown = typeof raw === "string" ? JSON.parse(raw) : raw;
  return Array.isArray(parsed) ? (parsed as TracedMessage[]) : [parsed as TracedMessage];
}

function contentsOf(span: CapturedSpan): string[] {
  return inputOf(span).map((m) => m.content ?? "");
}

function wireText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .map((part) => (part as { type?: string; text?: string }))
    .filter((part) => part.type === "text")
    .map((part) => part.text ?? "")
    .join("");
}

function normalizeWire(messages: OpenAiMessage[]): unknown[] {
  return messages
    .filter((m) => m.role !== "system")
    .map((m) => ({
      role: m.role,
      text: wireText(m.content),
      ...(m.tool_calls
        ? { calls: m.tool_calls.map((t) => `${t.function?.name}(${t.function?.arguments})`) }
        : {}),
      ...(m.tool_call_id ? { toolCallId: m.tool_call_id } : {}),
      ...(m.reasoning_content ? { reasoning: m.reasoning_content } : {}),
    }));
}

function normalizeTraced(messages: TracedMessage[]): unknown[] {
  return messages.map((m) => ({
    role: m.role,
    text: m.content ?? "",
    ...(m.tool_calls
      ? { calls: m.tool_calls.map((t) => `${t.function.name}(${t.function.arguments})`) }
      : {}),
    ...(m.tool_call_id ? { toolCallId: m.tool_call_id } : {}),
    ...(m.thinking ? { reasoning: m.thinking.map((t) => t.content).join("\n") } : {}),
  }));
}

describe("integration: pi -> extension -> Langfuse export", () => {
  let mock: MockProvider;

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

      // Trace fields on EVERY span, not only the root: Langfuse reads them per
      // span, and session-level aggregation groups by the session id on each
      // row, so the generations that carry the cost must carry it too.
      assert.equal(tree.length, 6, "root + 3 generations + 2 tools");
      for (const span of tree) {
        assert.equal(span.attrs["session.id"], root.attrs["session.id"], `${span.name} must carry the session id`);
        assert.equal(span.attrs["user.id"], "test-user", `${span.name} must carry the user id`);
        assert.deepEqual(span.attrs["langfuse.trace.tags"], ["pi"], `${span.name} must carry the tags`);
        assert.equal(
          span.attrs["langfuse.trace.name"],
          root.attrs["langfuse.trace.name"],
          `${span.name} must carry the trace name`,
        );
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

      assert.equal(
        subagentRoot!.attrs["session.id"],
        parentRoot!.attrs["session.id"],
        "subagent spans must join the parent's session",
      );
      assert.equal(subagentRoot!.attrs["user.id"], "test-user", "subagent spans must carry the user id");
      assert.deepEqual(subagentRoot!.attrs["langfuse.trace.tags"], ["pi"], "subagent spans must carry the tags");
      const meta = (k: string) => subagentRoot!.attrs[`langfuse.observation.metadata.${k}`];
      assert.equal(String(meta("pi_subagent")), "true");
      assert.equal(Number(meta("subagent_depth")), 1);

      // The model calls of the subagent go in the same trace. Their tokens add
      // to the parent totals.
      const subagentChildren = spans.filter((s) => s.parentSpanId === subagentRoot!.spanId);
      for (const child of subagentChildren) {
        assert.equal(
          child.attrs["session.id"],
          parentRoot!.attrs["session.id"],
          `${child.name} (subagent child) must carry the parent's session id`,
        );
        assert.equal(child.attrs["langfuse.trace.name"], undefined, `${child.name} must not carry a trace name`);
      }
      assert.ok(
        subagentChildren.some((s) => s.attrs["langfuse.observation.type"] === "generation"),
        "the subagent's generations must be nested under it",
      );

      const subagentGenerations = byStart(
        subagentChildren.filter((s) => s.name === "LLM Call"),
      );
      assert.ok(subagentGenerations.length >= 1, "the subagent must have at least one generation");
      for (const generation of subagentGenerations) {
        assert.equal(
          generation.attrs["langfuse.observation.metadata.input_source"],
          "context",
          "a session-less subagent run still gets its history from the context",
        );
        const history = inputOf(generation);
        assert.equal(history[0]!.role, "user");
        assert.match(String(history[0]!.content), /^Task: inspect the repository/);
        assert.ok(
          !contentsOf(generation).some((c) => c.includes("Delegate the repo inspection")),
          "the parent's prompt must not leak into the subagent's history",
        );
      }
      const parentGenerations = byStart(
        spans.filter((s) => s.name === "LLM Call" && s.parentSpanId === parentRoot!.spanId),
      );
      assert.ok(
        contentsOf(parentGenerations[0]!).some((c) => c.includes("Delegate the repo inspection")),
        "the parent keeps its own prompt in its own history",
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

  it("gives every generation the conversation up to that call, not only the delta", async () => {
    const capture = await startCaptureServer();
    try {
      const sandbox = createSandbox(mock.port);
      const result = await runPi(sandbox, "Explore this project and summarize it", {
        env: buildLangfuseEnv(capture),
      });
      assert.equal(result.status, 0, `pi failed: ${result.stderr}`);
      await waitForRequests(capture, 1);
      const generations = byStart(findSpansByName(capture.spans(), "LLM Call"));
      assert.equal(generations.length, 3);

      assert.deepEqual(
        generations.map((g) => inputOf(g).length),
        [1, 3, 5],
      );
      for (const generation of generations) {
        assert.equal(
          generation.attrs["langfuse.observation.metadata.input_source"],
          "context",
          "the history must come from pi's own context, never the delta fallback",
        );
      }
      assert.equal(
        Number(generations[2]!.attrs["langfuse.observation.metadata.history_message_count"]),
        5,
      );

      for (const generation of generations) {
        const first = inputOf(generation)[0]!;
        assert.equal(first.role, "user");
        assert.equal(first.content, "Explore this project and summarize it");
      }

      const last = inputOf(generations[2]!);
      assert.deepEqual(last.map((m) => m.role), ["user", "assistant", "tool", "assistant", "tool"]);
      assert.deepEqual(
        last.filter((m) => m.role === "tool").map((m) => m.name),
        ["bash", "read"],
      );
      assert.deepEqual(
        last.flatMap((m) => m.tool_calls?.map((t) => `${t.function.name} ${t.function.arguments}`) ?? []),
        ['bash {"command":"ls"}', 'read {"path":"README.md"}'],
      );
      for (const call of last.flatMap((m) => m.tool_calls ?? [])) {
        assert.equal(call.type, "function");
        assert.ok(call.id, "a tool call must keep its id");
      }

      const root = findSpansByName(capture.spans(), "Conversational Turn")[0]!;
      assert.deepEqual(JSON.parse(String(root.attrs["langfuse.observation.input"])), {
        role: "user",
        content: "Explore this project and summarize it",
      });
    } finally {
      capture.close();
    }
  });

  it("matches the conversation the provider actually received", async () => {
    const capture = await startCaptureServer();
    const firstCall = mock.sentMessages.length;
    try {
      const sandbox = createSandbox(mock.port);
      const result = await runPi(sandbox, "Explore this project and summarize it", {
        env: buildLangfuseEnv(capture),
      });
      assert.equal(result.status, 0, `pi failed: ${result.stderr}`);
      await waitForRequests(capture, 1);
      const generations = byStart(findSpansByName(capture.spans(), "LLM Call"));
      const sent = mock.sentMessages.slice(firstCall);
      assert.equal(sent.length, generations.length, "one provider request per generation");

      for (const [index, generation] of generations.entries()) {
        assert.deepEqual(
          normalizeTraced(inputOf(generation)),
          normalizeWire(sent[index]!),
          `generation ${index + 1} input must equal the provider request`,
        );
      }
    } finally {
      capture.close();
    }
  });

  it("keeps the history complete when the plugin restarts mid-session", async () => {
    const capture = await startCaptureServer();
    try {
      const sandbox = createSandbox(mock.port);
      const env = buildLangfuseEnv(capture);
      assert.equal((await runPi(sandbox, "First prompt", { env })).status, 0);
      assert.equal((await runPi(sandbox, "Second prompt", { env, continue: true })).status, 0);
      await waitForRequests(capture, 2);

      const roots = byStart(findSpansByName(capture.spans(), "Conversational Turn"));
      assert.equal(roots.length, 2);
      const secondTurn = byStart(
        findSpansByName(capture.spans(), "LLM Call").filter((s) => s.traceId === roots[1]!.traceId),
      );
      assert.equal(secondTurn.length, 3);

      const first = inputOf(secondTurn[0]!);
      assert.equal(first.length, 7, "turn 1 (5 messages) plus the new prompt and its first step");
      assert.equal(first[0]!.content, "First prompt", "turn 1's prompt must still be there");
      assert.deepEqual(
        contentsOf(secondTurn[0]!).filter((c) => c === "First prompt" || c === "Second prompt"),
        ["First prompt", "Second prompt"],
        "both turns appear, in order",
      );
      assert.equal(inputOf(secondTurn[2]!).length, 11);
      for (const generation of secondTurn) {
        assert.equal(generation.attrs["langfuse.observation.metadata.input_source"], "context");
      }
    } finally {
      capture.close();
    }
  });

  it("carries the compaction summary after a compaction, not the messages it replaced", async () => {
    const capture = await startCaptureServer();
    try {
      const sandbox = createSandbox(mock.port, {
        contextWindow: 18000,
        keepRecentTokens: 300,
        readmeFillerLines: 200,
      });
      const env = buildLangfuseEnv(capture);
      assert.equal((await runPi(sandbox, "Explore this project and summarize it", { env })).status, 0);
      assert.equal((await runPi(sandbox, "What did we conclude?", { env, continue: true })).status, 0);
      await waitForRequests(capture, 2);

      const spans = capture.spans();
      assert.equal(findSpansByName(spans, "Compaction").length > 0, true, "a compaction must have happened");
      const roots = byStart(findSpansByName(spans, "Conversational Turn"));
      assert.equal(roots.length, 2);

      const beforeCompaction = byStart(
        findSpansByName(spans, "LLM Call").filter((s) => s.traceId === roots[0]!.traceId),
      );
      const filler = "This paragraph is filler";
      assert.ok(
        contentsOf(beforeCompaction[2]!).some((c) => c.includes(filler)),
        "the pre-compaction turn does carry the long tool result",
      );

      const afterCompaction = byStart(
        findSpansByName(spans, "LLM Call").filter((s) => s.traceId === roots[1]!.traceId),
      );
      const first = inputOf(afterCompaction[0]!);
      assert.equal(first[0]!.role, "user");
      assert.match(
        first[0]!.content ?? "",
        /compacted into the following summary/,
        "the summary the model sees must be the first message",
      );
      assert.equal(
        first.length,
        3,
        "summary + kept entries + the new prompt, not the five messages the summary replaced",
      );
      assert.equal(first.at(-1)!.content, "What did we conclude?");
      assert.ok(
        !contentsOf(afterCompaction[0]!).some((c) => c.includes(filler)),
        "content the compaction dropped must not reappear in the history",
      );
    } finally {
      capture.close();
    }
  });

  it("attaches the assistant reasoning to the step that produced it", async () => {
    const capture = await startCaptureServer();
    const firstCall = mock.sentMessages.length;
    try {
      const sandbox = createSandbox(mock.port);
      const env = buildLangfuseEnv(capture);
      assert.equal((await runPi(sandbox, "Explore this project and summarize it", { env })).status, 0);
      assert.equal((await runPi(sandbox, "And the filler text?", { env, continue: true })).status, 0);
      await waitForRequests(capture, 2);

      const roots = byStart(findSpansByName(capture.spans(), "Conversational Turn"));
      const secondTurn = byStart(
        findSpansByName(capture.spans(), "LLM Call").filter((s) => s.traceId === roots[1]!.traceId),
      );
      const history = inputOf(secondTurn[0]!);

      const reasoning = history.filter((m) => m.thinking);
      assert.equal(reasoning.length, 1, "only the step that reasoned carries a thinking block");
      assert.deepEqual(reasoning[0]!.thinking, [{ type: "thinking", content: FINAL_ANSWER_THINKING }]);
      assert.equal(
        reasoning[0]!.content,
        "This is the test workspace. Done.",
        "the reasoning rides next to that step's text, not instead of it",
      );
      assert.ok(
        !contentsOf(secondTurn[0]!).some((c) => c.includes(FINAL_ANSWER_THINKING)),
        "reasoning must not be mixed into the message content",
      );

      const exported = JSON.stringify(capture.requests);
      assert.ok(!exported.includes("thinkingSignature"), "the signature blob must not be traced");
      assert.ok(!exported.includes("reasoning_content"), "the raw provider field must not be traced");

      const sent = mock.sentMessages.slice(firstCall);
      const withReasoning = sent.at(-1)!.filter((m) => m.reasoning_content);
      assert.equal(withReasoning.length, 1);
      assert.equal(withReasoning[0]!.reasoning_content, FINAL_ANSWER_THINKING);
    } finally {
      capture.close();
    }
  });
});
