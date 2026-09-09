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
  const plain = { reasoning: false };

  it("reads the cap the payload really carries, whatever the dialect calls it", () => {
    const payloads: Array<[string, unknown]> = [
      ["anthropic-messages / openai-completions", { max_tokens: 1477 }],
      ["openai-completions on newer OpenAI models", { max_completion_tokens: 1477 }],
      ["openai-responses / azure-openai-responses", { max_output_tokens: 1477 }],
      ["mistral-conversations", { maxTokens: 1477 }],
      ["google-generative-ai / google-vertex", { config: { maxOutputTokens: 1477 } }],
      ["bedrock-converse-stream", { inferenceConfig: { maxTokens: 1477 } }],
      ["pi-messages", { options: { maxTokens: 1477 } }],
    ];
    for (const [dialect, payload] of payloads) {
      assert.deepEqual(extractModelParameters(payload, plain), { max_tokens: 1477 }, dialect);
    }
  });

  it("reports the clamped cap, not the model's configured one", () => {
    assert.deepEqual(extractModelParameters({ max_tokens: 1477 }, { reasoning: false }), { max_tokens: 1477 });
  });

  it("adds the thinking level only for reasoning models", () => {
    assert.deepEqual(extractModelParameters({ max_tokens: 8192 }, { reasoning: true }, "high"), {
      max_tokens: 8192,
      thinking_level: "high",
    });
    assert.deepEqual(extractModelParameters({ max_tokens: 8192 }, plain, "high"), { max_tokens: 8192 });
  });

  it("treats `off` and a missing level as no thinking", () => {
    assert.deepEqual(extractModelParameters({ max_tokens: 4096 }, { reasoning: true }, "off"), { max_tokens: 4096 });
    assert.deepEqual(extractModelParameters({ max_tokens: 4096 }, { reasoning: true }), { max_tokens: 4096 });
  });

  it("returns undefined when nothing is worth a chip", () => {
    assert.equal(extractModelParameters(undefined, plain), undefined);
    assert.equal(extractModelParameters({}, plain), undefined);
    assert.equal(extractModelParameters("not an object", plain), undefined);
    assert.equal(extractModelParameters({ max_tokens: 0 }, plain), undefined);
    assert.equal(extractModelParameters({}, undefined), undefined);
  });

  it("keeps the cap when the context has no model", () => {
    assert.deepEqual(extractModelParameters({ max_tokens: 8192 }, undefined, "high"), { max_tokens: 8192 });
  });

  it("keeps the thinking level when the payload has no cap", () => {
    assert.deepEqual(extractModelParameters({}, { reasoning: true }, "medium"), { thinking_level: "medium" });
  });

  it("rejects a cap that is not a positive whole number", () => {
    for (const bad of [Number.NaN, Number.POSITIVE_INFINITY, -5, 1.5, "8192", null, true]) {
      assert.equal(extractModelParameters({ max_tokens: bad }, plain), undefined, String(bad));
    }
  });

  it("never throws on a hostile payload", () => {
    const hostile: Record<string, unknown> = {};
    Object.defineProperty(hostile, "max_tokens", {
      enumerable: true,
      get() {
        throw new Error("payload getter exploded");
      },
    });
    assert.equal(extractModelParameters(hostile, plain), undefined);
    assert.equal(extractModelParameters(hostile, { reasoning: true }, "high"), undefined);
  });

  it("never descends into message arrays looking for a cap", () => {
    const payload = {
      messages: [{ role: "user", content: "budget_tokens: 99, max_tokens: 99" }, { max_tokens: 99 }],
      max_tokens: 1477,
    };
    assert.deepEqual(extractModelParameters(payload, plain), { max_tokens: 1477 });
    assert.deepEqual(extractModelParameters({ messages: [{ max_tokens: 99 }] }, plain), undefined);
  });
});

function langfuseEnv(capture: Capture): Record<string, string> {
  return {
    LANGFUSE_PUBLIC_KEY: "pk-lf-test",
    LANGFUSE_SECRET_KEY: "sk-lf-test",
    LANGFUSE_BASE_URL: `http://127.0.0.1:${capture.port}`,
  };
}

function generationParameters(spans: CapturedSpan[]): Array<Record<string, unknown> | undefined> {
  return spans
    .filter((s) => s.name === "LLM Call")
    .sort((a, b) => (a.startNs < b.startNs ? -1 : 1))
    .map((s) => {
      const raw = s.attrs["langfuse.observation.model.parameters"];
      return raw === undefined ? undefined : (JSON.parse(String(raw)) as Record<string, unknown>);
    });
}

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

  it("reports the cap pi clamped to the context, not the model's 8192", async () => {
    const capture = await startCaptureServer();
    try {
      const sandbox = createSandbox(mock.port, { contextWindow: 18000, readmeFillerLines: 700 });
      writeFileSync(join(sandbox.agentDir, "settings.json"), JSON.stringify({ compaction: { enabled: false } }));
      const result = await runPi(sandbox, "Explore this project and summarize it", { env: langfuseEnv(capture) });
      assert.equal(result.status, 0, `pi failed: ${result.stderr}`);
      await waitForRequests(capture, 1);

      const caps = generationParameters(capture.spans()).map((p) => p?.max_tokens as number);
      assert.equal(caps.length, 3);
      assert.deepEqual(caps.slice(0, 2), [8192, 8192], "the early calls still fit the configured cap");
      const clamped = caps[2]!;
      assert.ok(clamped > 0 && clamped < 8192, `the last call must report the clamped cap, got ${clamped}`);
    } finally {
      capture.close();
    }
  });
});
