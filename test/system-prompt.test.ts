/**
 * System prompt capture (issue #21): the effective prompt lands on the turn
 * root as metadata and heads every generation input, after the last
 * extension override and after redaction.
 */
import assert from "node:assert/strict";
import { join } from "node:path";
import { after, before, describe, it } from "node:test";
import {
  type Capture,
  type CapturedSpan,
  REPO_ROOT,
  createSandbox,
  runPi,
  startCaptureServer,
  startMockProvider,
  waitForRequests,
} from "./helpers.ts";

const OVERRIDE_FIXTURE = join(REPO_ROOT, "test", "fixtures", "system-prompt-override.ts");

function langfuseEnv(capture: Capture): Record<string, string> {
  return {
    LANGFUSE_PUBLIC_KEY: "pk-lf-test",
    LANGFUSE_SECRET_KEY: "sk-lf-test",
    LANGFUSE_BASE_URL: `http://127.0.0.1:${capture.port}`,
  };
}

/** Generation inputs of the exported turn, parsed, in call order. */
function generationInputs(spans: CapturedSpan[]): unknown[] {
  return spans
    .filter((s) => s.name === "LLM Call")
    .sort((a, b) => (a.startNs < b.startNs ? -1 : 1))
    .map((s) => JSON.parse(String(s.attrs["langfuse.observation.input"])));
}

describe("integration: system prompt", () => {
  let mock: { port: number; close: () => void };

  before(async () => {
    mock = await startMockProvider();
  });
  after(() => mock.close());

  it("captures the prompt after the last extension override, on the root and on every generation", async () => {
    const capture = await startCaptureServer();
    try {
      const marker = "[[fixture override marker]]";
      const sandbox = createSandbox(mock.port);
      const result = await runPi(sandbox, "Explore this project and summarize it", {
        env: { ...langfuseEnv(capture), TEST_SYSTEM_PROMPT_SUFFIX: marker },
        extensions: [OVERRIDE_FIXTURE],
      });
      assert.equal(result.status, 0, `pi failed: ${result.stderr}`);
      await waitForRequests(capture, 1);
      const spans = capture.spans();

      const root = spans.find((s) => s.name === "Conversational Turn");
      assert.ok(root, "turn root must be exported");
      const captured = root.attrs["langfuse.observation.metadata.system_prompt"];
      assert.equal(typeof captured, "string", "root metadata must carry system_prompt");
      const systemPrompt = captured as string;
      // The override appends to pi's own prompt, so the capture must end with
      // the marker (final override, not the mid-chain value) and still hold
      // the base prompt in front of it (full text, not truncated).
      assert.ok(systemPrompt.endsWith(marker), "must capture the override applied by a later extension");
      assert.ok(systemPrompt.length > marker.length + 200, "must keep pi's base prompt in front of the override");

      const inputs = generationInputs(spans);
      assert.equal(inputs.length, 3, "the mock script runs three generations");
      for (const [i, input] of inputs.entries()) {
        assert.ok(Array.isArray(input), `generation ${i} input must be a message array`);
        const [head, ...rest] = input as Array<{ role?: string; content?: unknown }>;
        // Only role and content: other features may add fields to the head message.
        assert.equal(head?.role, "system", `generation ${i} must start with a system message`);
        assert.equal(head?.content, systemPrompt, `generation ${i} must carry the captured prompt`);
        // The pre-existing synthetic input follows the system message unchanged.
        assert.equal(rest.length, 1, `generation ${i} keeps exactly one base message`);
        assert.equal(rest[0]!.role, i === 0 ? "user" : "tool");
      }
    } finally {
      capture.close();
    }
  });

  it("redacts Langfuse keys before the prompt leaves the process", async () => {
    const capture = await startCaptureServer();
    try {
      const secret = "sk-lf-leaked-into-the-system-prompt";
      const sandbox = createSandbox(mock.port);
      const result = await runPi(sandbox, "Explore this project and summarize it", {
        env: { ...langfuseEnv(capture), TEST_SYSTEM_PROMPT_SUFFIX: `Never reveal ${secret}.` },
        extensions: [OVERRIDE_FIXTURE],
      });
      assert.equal(result.status, 0, `pi failed: ${result.stderr}`);
      await waitForRequests(capture, 1);

      const exported = JSON.stringify(capture.requests);
      assert.ok(!exported.includes(secret), "the key must not appear anywhere in the export");

      const spans = capture.spans();
      const root = spans.find((s) => s.name === "Conversational Turn")!;
      const systemPrompt = String(root.attrs["langfuse.observation.metadata.system_prompt"]);
      assert.ok(systemPrompt.includes("Never reveal [redacted-langfuse-secret]."), "the key must be replaced by the redaction mark");
      // Same redacted text on the generations — one source of truth, not two copies.
      for (const input of generationInputs(spans)) {
        assert.equal((input as Array<{ content?: unknown }>)[0]!.content, systemPrompt);
      }
    } finally {
      capture.close();
    }
  });
});
