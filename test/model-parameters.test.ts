/**
 * Model parameters on generations: the unit half pins the mapping from pi's
 * model/thinking level to the chip record, the integration half checks the
 * exported `langfuse.observation.model.parameters` attribute for a plain and
 * a reasoning model.
 */
import assert from "node:assert/strict";
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { after, before, describe, it } from "node:test";
import { extractModelParameters } from "../src/index.ts";
import {
  type Capture,
  type CapturedSpan,
  type Sandbox,
  createSandbox,
  runPi,
  startCaptureServer,
  startMockProvider,
  waitForRequests,
} from "./helpers.ts";

describe("extractModelParameters", () => {
  it("returns undefined without a model", () => {
    assert.equal(extractModelParameters(undefined, "high"), undefined);
  });

  it("reports the configured max tokens", () => {
    assert.deepEqual(extractModelParameters({ maxTokens: 8192, reasoning: false }), { max_tokens: 8192 });
  });

  it("adds the thinking level only for reasoning models", () => {
    assert.deepEqual(extractModelParameters({ maxTokens: 8192, reasoning: true }, "high"), {
      max_tokens: 8192,
      thinking_level: "high",
    });
    // pi clamps the level to "off" for non-reasoning models before sending;
    // a stale level on the context must not produce a misleading chip.
    assert.deepEqual(extractModelParameters({ maxTokens: 8192, reasoning: false }, "high"), { max_tokens: 8192 });
  });

  it("treats `off` and a missing level as no thinking", () => {
    assert.deepEqual(extractModelParameters({ maxTokens: 4096, reasoning: true }, "off"), { max_tokens: 4096 });
    assert.deepEqual(extractModelParameters({ maxTokens: 4096, reasoning: true }), { max_tokens: 4096 });
  });

  it("returns undefined when nothing is worth a chip", () => {
    // pi's Agent starts with maxTokens: 0 before a model is configured.
    assert.equal(extractModelParameters({ maxTokens: 0, reasoning: false }), undefined);
    assert.equal(extractModelParameters({}), undefined);
    assert.equal(extractModelParameters({ reasoning: true }, "off"), undefined);
  });

  it("keeps the thinking level when max tokens is unknown", () => {
    assert.deepEqual(extractModelParameters({ reasoning: true }, "medium"), { thinking_level: "medium" });
  });
});

function langfuseEnv(capture: Capture): Record<string, string> {
  return {
    LANGFUSE_PUBLIC_KEY: "pk-lf-test",
    LANGFUSE_SECRET_KEY: "sk-lf-test",
    LANGFUSE_BASE_URL: `http://127.0.0.1:${capture.port}`,
  };
}

/** Parsed model parameters of every exported generation, in call order. */
function generationParameters(spans: CapturedSpan[]): Array<Record<string, unknown> | undefined> {
  return spans
    .filter((s) => s.name === "LLM Call")
    .sort((a, b) => (a.startNs < b.startNs ? -1 : 1))
    .map((s) => {
      const raw = s.attrs["langfuse.observation.model.parameters"];
      return raw === undefined ? undefined : (JSON.parse(String(raw)) as Record<string, unknown>);
    });
}

/**
 * Turn the sandbox's mock model into a reasoning model and pick a default
 * thinking level. pi clamps the level to what the model supports, so the
 * model flag and the setting are both needed for the level to survive.
 */
function enableReasoning(sandbox: Sandbox, level: string) {
  const modelsPath = join(sandbox.agentDir, "models.json");
  const models = JSON.parse(readFileSync(modelsPath, "utf8"));
  models.providers.mock.models[0].reasoning = true;
  writeFileSync(modelsPath, JSON.stringify(models));
  writeFileSync(join(sandbox.agentDir, "settings.json"), JSON.stringify({ defaultThinkingLevel: level }));
}

describe("integration: model parameters", () => {
  let mock: { port: number; close: () => void };

  before(async () => {
    mock = await startMockProvider();
  });
  after(() => mock.close());

  it("records max tokens on every generation of a plain model", async () => {
    const capture = await startCaptureServer();
    try {
      const sandbox = createSandbox(mock.port);
      const result = await runPi(sandbox, "Explore this project and summarize it", { env: langfuseEnv(capture) });
      assert.equal(result.status, 0, `pi failed: ${result.stderr}`);
      await waitForRequests(capture, 1);

      const params = generationParameters(capture.spans());
      assert.equal(params.length, 3, "the mock script runs three generations");
      for (const p of params) {
        // The sandbox model declares maxTokens: 8192 and reasoning: false.
        assert.deepEqual(p, { max_tokens: 8192 });
      }
    } finally {
      capture.close();
    }
  });

  it("adds the thinking level for a reasoning model", async () => {
    const capture = await startCaptureServer();
    try {
      const sandbox = createSandbox(mock.port);
      enableReasoning(sandbox, "high");
      const result = await runPi(sandbox, "Explore this project and summarize it", { env: langfuseEnv(capture) });
      assert.equal(result.status, 0, `pi failed: ${result.stderr}`);
      await waitForRequests(capture, 1);

      const params = generationParameters(capture.spans());
      assert.equal(params.length, 3);
      for (const p of params) {
        assert.deepEqual(p, { max_tokens: 8192, thinking_level: "high" });
      }
    } finally {
      capture.close();
    }
  });
});
