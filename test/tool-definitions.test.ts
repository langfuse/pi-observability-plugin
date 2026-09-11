import assert from "node:assert/strict";
import { after, before, describe, it } from "node:test";
import { activeToolDefinitions, attachToolDefinitions, type ToolDefinitionInput } from "../src/index.ts";
import {
  type Capture,
  type CapturedSpan,
  createSandbox,
  runPi,
  startCaptureServer,
  startMockProvider,
  waitForRequests,
} from "./helpers.ts";

const TOOLS: ToolDefinitionInput[] = [{ name: "read", description: "Read a file", parameters: { type: "object" } }];

describe("attachToolDefinitions", () => {
  it("adds the tools to the first message without mutating the caller's data", () => {
    const input = [
      { role: "system", content: "be brief" },
      { role: "user", content: "hi" },
    ];
    const out = attachToolDefinitions(input, TOOLS) as Array<Record<string, unknown>>;
    assert.deepEqual(out, [{ role: "system", content: "be brief", tools: TOOLS }, { role: "user", content: "hi" }]);
    assert.equal("tools" in input[0]!, false);
  });

  it("passes the input through when there is nothing to attach or nothing to attach to", () => {
    const message = [{ role: "user", content: "hi" }];
    assert.equal(attachToolDefinitions(message, []), message);
    assert.equal(attachToolDefinitions(undefined, TOOLS), undefined);
  });
});

describe("activeToolDefinitions", () => {
  const registry = [
    { name: "read", description: "Read a file", parameters: { type: "object" }, promptGuidelines: ["x"], sourceInfo: {} },
    { name: "bash", description: "Run a command", parameters: { type: "object" }, sourceInfo: {} },
    { name: "grep", description: "Search files", parameters: { type: "object" }, sourceInfo: {} },
  ];

  it("keeps only the active tools, in the active order, with only the ChatML fields", () => {
    const pi = { getAllTools: () => registry as never, getActiveTools: () => ["bash", "read"] };
    assert.deepEqual(activeToolDefinitions(pi), [
      { name: "bash", description: "Run a command", parameters: { type: "object" } },
      { name: "read", description: "Read a file", parameters: { type: "object" } },
    ]);
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

function generationInputs(spans: CapturedSpan[]): unknown[] {
  return spans
    .filter((s) => s.name === "LLM Call")
    .sort((a, b) => (a.startNs < b.startNs ? -1 : 1))
    .map((s) => JSON.parse(String(s.attrs["langfuse.observation.input"])));
}

describe("integration: tool definitions", () => {
  let mock: { port: number; close: () => void };

  before(async () => {
    mock = await startMockProvider();
  });
  after(() => mock.close());

  it("attaches the active tools to every generation, as a ChatML message array", async () => {
    const capture = await startCaptureServer();
    try {
      const sandbox = createSandbox(mock.port);
      const result = await runPi(sandbox, "Explore this project and summarize it", { env: langfuseEnv(capture) });
      assert.equal(result.status, 0, `pi failed: ${result.stderr}`);
      await waitForRequests(capture, 1);

      const inputs = generationInputs(capture.spans());
      assert.equal(inputs.length, 3, "the mock script runs three generations");
      for (const [i, input] of inputs.entries()) {
        assert.ok(Array.isArray(input), `generation ${i} input must be an array of messages, got ${typeof input}`);
        const [head, ...rest] = input as Array<Record<string, unknown>>;
        assert.equal(head?.role, "system", `generation ${i} must start with the system message`);
        assert.equal(rest.length, 1, `generation ${i} keeps exactly one base message, not a nested array`);
        assert.equal(rest[0]!.role, i === 0 ? "user" : "tool");
        const tools = head!.tools as ToolDefinitionInput[];
        assert.ok(Array.isArray(tools), `generation ${i} must carry tools on its first message`);
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
});
