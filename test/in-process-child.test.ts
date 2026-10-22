/**
 * In-process subagents: an orchestrator extension runs the child agent as a
 * second AgentSession in the same pi process, which loads this extension again
 * and disposes it with session_shutdown "quit". The parent's spans must keep
 * exporting after that, and each child must nest under the turn that spawned
 * it, not under a sibling.
 */
import assert from "node:assert/strict";
import { join } from "node:path";
import { after, before, describe, it } from "node:test";
import {
  type Capture,
  type CapturedSpan,
  type MockProvider,
  REPO_ROOT,
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
  };
}

function findSpansByName(spans: CapturedSpan[], name: string): CapturedSpan[] {
  return spans.filter((s) => s.name === name);
}

describe("integration: in-process subagents", () => {
  let mock: MockProvider;

  before(async () => {
    mock = await startMockProvider();
  });

  after(async () => {
    mock.close();
  });

  it("keeps tracing the parent after an in-process child is disposed", async () => {
    const capture = await startCaptureServer();
    try {
      const sandbox = createSandbox(mock.port);
      const result = await runPi(sandbox, "Delegate the repo inspection", {
        env: buildLangfuseEnv(capture),
        extensions: [join(REPO_ROOT, "test", "fixtures", "in-process-subagent-tool.ts")],
      });
      assert.equal(result.status, 0, `pi failed: ${result.stderr}`);
      await waitForRequests(capture, 2, 15_000);
      const spans = capture.spans();

      const parentRoot = findSpansByName(spans, "Conversational Turn")[0];
      const subagentRoot = findSpansByName(spans, "Subagent Turn")[0];
      assert.ok(parentRoot, "parent turn must be traced");
      assert.ok(subagentRoot, "subagent turn must be traced");

      // The parent's second generation starts after the child was disposed.
      const parentGenerations = spans.filter(
        (s) => s.name === "LLM Call" && s.parentSpanId === parentRoot!.spanId,
      );
      assert.equal(parentGenerations.length, 2, "the generation after the child's dispose must still export");
      assert.ok(
        spans.some((s) => s.name === "Tool: subagent" && s.parentSpanId === parentRoot!.spanId),
        "the delegating tool call must be traced",
      );

      assert.equal(subagentRoot!.traceId, parentRoot!.traceId, "subagent must join the parent's trace");
      assert.equal(
        subagentRoot!.parentSpanId,
        parentRoot!.spanId,
        "subagent root must be a child of the parent turn root",
      );
    } finally {
      capture.close();
    }
  });

  it("attaches parallel in-process children to the parent turn root, not to each other", async () => {
    const capture = await startCaptureServer();
    try {
      const sandbox = createSandbox(mock.port);
      const result = await runPi(sandbox, "Delegate the repo inspection", {
        env: { ...buildLangfuseEnv(capture), TEST_SUBAGENT_OVERLAP: "1" },
        extensions: [join(REPO_ROOT, "test", "fixtures", "in-process-subagent-tool.ts")],
      });
      assert.equal(result.status, 0, `pi failed: ${result.stderr}`);
      await waitForRequests(capture, 3, 15_000);
      const spans = capture.spans();

      const parentRoot = findSpansByName(spans, "Conversational Turn")[0];
      assert.ok(parentRoot, "parent turn must be traced");
      const subagentRoots = findSpansByName(spans, "Subagent Turn");
      assert.equal(subagentRoots.length, 2, "two parallel children must both be traced");

      for (const child of subagentRoots) {
        assert.equal(child.traceId, parentRoot!.traceId, "each subagent must join the parent's trace");
        assert.equal(
          child.parentSpanId,
          parentRoot!.spanId,
          "each subagent root must be a child of the parent turn root, not of a sibling",
        );
      }
    } finally {
      capture.close();
    }
  });
});
