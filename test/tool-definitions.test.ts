/**
 * Tool definitions on generations: the unit half pins the pure helpers, the
 * integration half checks that the exported generation input carries the
 * tools pi actually offers — extension tools included, deactivated ones not.
 */
import assert from "node:assert/strict";
import { join } from "node:path";
import { after, before, describe, it } from "node:test";
import { activeToolDefinitions, attachToolDefinitions, type ToolDefinitionInput } from "../src/index.ts";
import { PROBE_ACTIVE_TOOL, PROBE_INACTIVE_TOOL } from "./fixtures/probe-tools.ts";
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

const PROBE_FIXTURE = join(REPO_ROOT, "test", "fixtures", "probe-tools.ts");

const TOOLS: ToolDefinitionInput[] = [{ name: "read", description: "Read a file", parameters: { type: "object" } }];

describe("attachToolDefinitions", () => {
  it("leaves the input alone when there is nothing to attach or nothing to attach to", () => {
    const message = { role: "user", content: "hi" };
    assert.equal(attachToolDefinitions(message, []), message);
    assert.equal(attachToolDefinitions(undefined, TOOLS), undefined);
    assert.equal(attachToolDefinitions("plain text", TOOLS), "plain text");
  });

  it("adds the tools to a single message", () => {
    assert.deepEqual(attachToolDefinitions({ role: "user", content: "hi" }, TOOLS), {
      role: "user",
      content: "hi",
      tools: TOOLS,
    });
  });

  it("adds the tools to the first message of an array only", () => {
    const input = [
      { role: "system", content: "be brief" },
      { role: "user", content: "hi" },
    ];
    const out = attachToolDefinitions(input, TOOLS) as Array<Record<string, unknown>>;
    assert.deepEqual(out, [{ role: "system", content: "be brief", tools: TOOLS }, { role: "user", content: "hi" }]);
    assert.equal(out[1], input[1], "later messages are passed through by reference");
    assert.equal("tools" in input[0]!, false, "the caller's array is not mutated");
  });

  it("skips arrays whose head is not a message", () => {
    const input = ["not a message", { role: "user", content: "hi" }];
    assert.equal(attachToolDefinitions(input, TOOLS), input);
    assert.deepEqual(attachToolDefinitions([], TOOLS), []);
  });
});

describe("activeToolDefinitions", () => {
  const registry = [
    { name: "read", description: "Read a file", parameters: { type: "object" }, promptGuidelines: ["x"], sourceInfo: {} },
    { name: "bash", description: "Run a command", parameters: { type: "object" }, sourceInfo: {} },
    { name: "mcp_search", description: "Search", parameters: { type: "object" }, sourceInfo: {} },
  ];

  it("narrows the registry to the active names and keeps only the ChatML fields", () => {
    const pi = { getAllTools: () => registry as never, getActiveTools: () => ["bash", "read"] };
    assert.deepEqual(activeToolDefinitions(pi), [
      { name: "read", description: "Read a file", parameters: { type: "object" } },
      { name: "bash", description: "Run a command", parameters: { type: "object" } },
    ]);
  });

  it("returns nothing when no tool is active", () => {
    const pi = { getAllTools: () => registry as never, getActiveTools: () => [] };
    assert.deepEqual(activeToolDefinitions(pi), []);
  });

  it("degrades to nothing on a pi without the accessors or with throwing ones", () => {
    assert.deepEqual(activeToolDefinitions({} as never), []);
    const throwing = {
      getAllTools: () => {
        throw new Error("not ready");
      },
      getActiveTools: () => ["read"],
    };
    assert.deepEqual(activeToolDefinitions(throwing as never), []);
  });
});

function langfuseEnv(capture: Capture): Record<string, string> {
  return {
    LANGFUSE_PUBLIC_KEY: "pk-lf-test",
    LANGFUSE_SECRET_KEY: "sk-lf-test",
    LANGFUSE_BASE_URL: `http://127.0.0.1:${capture.port}`,
  };
}

/** Parsed generation inputs of the exported turn, in call order. */
function generationInputs(spans: CapturedSpan[]): unknown[] {
  return spans
    .filter((s) => s.name === "LLM Call")
    .sort((a, b) => (a.startNs < b.startNs ? -1 : 1))
    .map((s) => JSON.parse(String(s.attrs["langfuse.observation.input"])));
}

/** The message that carries `tools`, for a single-message or array input. */
function firstMessage(input: unknown): Record<string, unknown> {
  const head = Array.isArray(input) ? input[0] : input;
  assert.ok(head && typeof head === "object", "generation input must hold a message");
  return head as Record<string, unknown>;
}

describe("integration: tool definitions", () => {
  let mock: { port: number; close: () => void };

  before(async () => {
    mock = await startMockProvider();
  });
  after(() => mock.close());

  it("attaches the built-in tools to every generation", async () => {
    const capture = await startCaptureServer();
    try {
      const sandbox = createSandbox(mock.port);
      const result = await runPi(sandbox, "Explore this project and summarize it", { env: langfuseEnv(capture) });
      assert.equal(result.status, 0, `pi failed: ${result.stderr}`);
      await waitForRequests(capture, 1);

      const inputs = generationInputs(capture.spans());
      assert.equal(inputs.length, 3, "the mock script runs three generations");
      for (const [i, input] of inputs.entries()) {
        const tools = firstMessage(input).tools as ToolDefinitionInput[];
        assert.ok(Array.isArray(tools), `generation ${i} must carry tools on its first message`);
        // pi's default active set, in pi's order.
        assert.deepEqual(
          tools.map((t) => t.name),
          ["read", "bash", "edit", "write"],
        );
        for (const tool of tools) {
          assert.equal(typeof tool.description, "string");
          assert.equal((tool.parameters as { type?: unknown })?.type, "object", `${tool.name} keeps its JSON schema`);
        }
      }
    } finally {
      capture.close();
    }
  });

  it("includes extension tools and leaves deactivated ones out", async () => {
    const capture = await startCaptureServer();
    try {
      const sandbox = createSandbox(mock.port);
      const result = await runPi(sandbox, "Explore this project and summarize it", {
        env: langfuseEnv(capture),
        extensions: [PROBE_FIXTURE],
      });
      assert.equal(result.status, 0, `pi failed: ${result.stderr}`);
      await waitForRequests(capture, 1);

      const inputs = generationInputs(capture.spans());
      assert.equal(inputs.length, 3);
      for (const input of inputs) {
        const tools = firstMessage(input).tools as ToolDefinitionInput[];
        const byName = new Map(tools.map((t) => [t.name, t]));
        assert.ok(byName.has("read"), "built-ins stay");
        assert.ok(!byName.has(PROBE_INACTIVE_TOOL.name), "a registered but deactivated tool is not in the request");
        // The extension tool arrives as the ChatML definition, schema intact.
        assert.deepEqual(byName.get(PROBE_ACTIVE_TOOL.name), PROBE_ACTIVE_TOOL);
      }
    } finally {
      capture.close();
    }
  });
});
