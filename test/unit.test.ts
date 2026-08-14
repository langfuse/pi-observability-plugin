import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  extractText,
  buildCostDetails,
  maskSecrets,
  shortSessionLabel,
  extractToolCalls,
  truncateText,
  buildUsageDetails,
  type PiUsage,
} from "../src/index.ts";

describe("truncateText", () => {
  it("keeps short text untouched and records the original length", () => {
    const { text, meta } = truncateText("hello");
    assert.equal(text, "hello");
    assert.deepEqual(meta, { truncated: false, orig_len: 5 });
  });

  it("head-truncates long text and records kept_len + sha256 of the full text", () => {
    const long = "x".repeat(30_000);
    const { text, meta } = truncateText(long);
    assert.equal(text.length, 20_000);
    assert.equal(meta.truncated, true);
    assert.equal(meta.orig_len, 30_000);
    assert.equal(meta.kept_len, 20_000);
    assert.match(meta.sha256 ?? "", /^[a-f0-9]{64}$/);
  });
});

describe("maskSecrets", () => {
  it("masks Langfuse key patterns in plain strings", () => {
    const masked = maskSecrets("key is sk-lf-abc123 and pk-lf-def456", []);
    assert.equal(masked, "key is [LANGFUSE_KEY_REDACTED] and [LANGFUSE_KEY_REDACTED]");
  });

  it("masks configured literal secrets", () => {
    const masked = maskSecrets("token: super-secret-token", ["super-secret-token"]);
    assert.equal(masked, "token: [LANGFUSE_KEY_REDACTED]");
  });

  it("recurses through nested objects and arrays", () => {
    const masked = maskSecrets({ a: ["sk-lf-x1"], b: { c: "pk-lf-y2" } }, []) as {
      a: string[];
      b: { c: string };
    };
    assert.equal(masked.a[0], "[LANGFUSE_KEY_REDACTED]");
    assert.equal(masked.b.c, "[LANGFUSE_KEY_REDACTED]");
  });

  it("survives circular structures", () => {
    const obj: Record<string, unknown> = { name: "sk-lf-zzz" };
    obj.self = obj;
    const masked = maskSecrets(obj, []) as { name: string; self: unknown };
    assert.equal(masked.name, "[LANGFUSE_KEY_REDACTED]");
    assert.equal(masked.self, "[circular]");
  });

  it("does not falsely flag the same object in two branches as circular", () => {
    const shared = { v: "ok" };
    const masked = maskSecrets({ a: shared, b: shared }, []) as { a: { v: string }; b: { v: string } };
    assert.equal(masked.a.v, "ok");
    assert.equal(masked.b.v, "ok");
  });

  it("leaves non-strings untouched", () => {
    assert.deepEqual(maskSecrets({ n: 42, b: true, x: null }, []), { n: 42, b: true, x: null });
  });
});

describe("extractText / extractToolCalls", () => {
  const content = [
    { type: "text", text: "Hello " },
    { type: "thinking", thinking: "hmm" },
    { type: "toolCall", id: "call_1", name: "bash", arguments: { command: "ls" } },
    { type: "text", text: "world" },
  ];

  it("joins only text parts", () => {
    assert.equal(extractText(content), "Hello world");
  });

  it("accepts plain strings and rejects non-arrays", () => {
    assert.equal(extractText("plain"), "plain");
    assert.equal(extractText(undefined), "");
    assert.equal(extractText({}), "");
  });

  it("extracts tool calls as {id, name} only (no arguments)", () => {
    assert.deepEqual(extractToolCalls(content), [{ id: "call_1", name: "bash" }]);
  });
});

describe("shortSessionLabel", () => {
  it("uses the first block of a UUID", () => {
    assert.equal(shortSessionLabel("019fdc53-ba3c-73be-855b-b134fe182bbf"), "019fdc53");
  });

  it("falls back to the first 12 chars for other formats", () => {
    assert.equal(shortSessionLabel("2026-08-07T13-05-02"), "2026-08-07T1");
  });

  it("handles empty ids", () => {
    assert.equal(shortSessionLabel(""), "unknown");
  });
});

describe("buildUsageDetails / buildCostDetails", () => {
  const usage: PiUsage = {
    input: 2,
    output: 264,
    cacheRead: 2267,
    cacheWrite: 36,
    cost: { input: 0.00001, output: 0.0066, cacheRead: 0.0011335, cacheWrite: 0.000225, total: 0.0079685 },
  };

  it("maps pi usage to canonical Langfuse keys (cache split, no double counting)", () => {
    assert.deepEqual(buildUsageDetails(usage), {
      input: 2,
      output: 264,
      cache_read_input_tokens: 2267,
      cache_creation_input_tokens: 36,
    });
  });

  it("omits zero values", () => {
    assert.deepEqual(buildUsageDetails({ input: 10, output: 0, cacheRead: 0, cacheWrite: 0 }), { input: 10 });
  });

  it("returns undefined for all-zero usage", () => {
    assert.equal(buildUsageDetails({ input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }), undefined);
  });

  it("maps cost incl. cache buckets", () => {
    assert.deepEqual(buildCostDetails(usage), {
      total: 0.0079685,
      input: 0.00001,
      output: 0.0066,
      cache_read_input_tokens: 0.0011335,
      cache_creation_input_tokens: 0.000225,
    });
  });

  it("keeps cost keys mirroring usage keys so Langfuse can join them", () => {
    const usageKeys = Object.keys(buildUsageDetails(usage) ?? {});
    const costKeys = Object.keys(buildCostDetails(usage) ?? {}).filter((k) => k !== "total");
    for (const key of costKeys) {
      assert.ok(usageKeys.includes(key), `cost key "${key}" has no matching usage key`);
    }
  });

  it("omits cost details entirely when pi has no pricing (server-side pricing takes over)", () => {
    assert.equal(buildCostDetails({ input: 1, output: 1, cacheRead: 0, cacheWrite: 0 }), undefined);
    assert.equal(
      buildCostDetails({
        input: 1,
        output: 1,
        cacheRead: 0,
        cacheWrite: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      }),
      undefined,
    );
  });

  it("splits reasoning tokens out of output so the buckets stay exclusive", () => {
    assert.deepEqual(
      buildUsageDetails({ input: 500, output: 300, cacheRead: 0, cacheWrite: 0, reasoning: 200 }),
      { input: 500, output: 100, output_reasoning_tokens: 200 },
    );
  });

  it("keeps output intact when a provider reports inconsistent reasoning counts", () => {
    assert.deepEqual(
      buildUsageDetails({ input: 10, output: 5, cacheRead: 0, cacheWrite: 0, reasoning: 9 }),
      { input: 10, output: 5 },
    );
  });

  it("ignores a zero reasoning count from providers that always report the field", () => {
    assert.deepEqual(
      buildUsageDetails({ input: 10, output: 20, cacheRead: 0, cacheWrite: 0, reasoning: 0 }),
      { input: 10, output: 20 },
    );
  });

  it("emits only the reasoning bucket when all output tokens are reasoning", () => {
    // A model can think and write no text. Then the result has no `output` key.
    assert.deepEqual(
      buildUsageDetails({ input: 10, output: 50, cacheRead: 0, cacheWrite: 0, reasoning: 50 }),
      { input: 10, output_reasoning_tokens: 50 },
    );
  });
});
