import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  extractText,
  readInheritedParent,
  buildCostDetails,
  describeImage,
  extractImages,
  createSecretRedactor,
  renderContentWithImageMarkers,
  extractToolCalls,
  toMultimodalContent,
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

describe("createSecretRedactor", () => {
  it("redacts Langfuse key tokens in plain strings", () => {
    const redact = createSecretRedactor();
    assert.equal(
      redact("key is sk-lf-abc123 and pk-lf-def456"),
      "key is [redacted-langfuse-secret] and [redacted-langfuse-secret]",
    );
  });

  it("redacts configured literal secrets", () => {
    const redact = createSecretRedactor("super-secret-token");
    assert.equal(redact("token: super-secret-token"), "token: [redacted-langfuse-secret]");
  });

  it("escapes regex metacharacters in configured secrets", () => {
    const redact = createSecretRedactor("we+rd(secret)$");
    assert.equal(redact("x we+rd(secret)$ y"), "x [redacted-langfuse-secret] y");
  });

  it("recurses through nested objects and arrays", () => {
    const redact = createSecretRedactor();
    const redacted = redact({ a: ["sk-lf-x1"], b: { c: "pk-lf-y2" } }) as {
      a: string[];
      b: { c: string };
    };
    assert.equal(redacted.a[0], "[redacted-langfuse-secret]");
    assert.equal(redacted.b.c, "[redacted-langfuse-secret]");
  });

  it("collapses circular structures to a marker", () => {
    const obj: Record<string, unknown> = { name: "sk-lf-zzz" };
    obj.self = obj;
    const redact = createSecretRedactor();
    const redacted = redact(obj) as { name: string; self: unknown };
    assert.equal(redacted.name, "[redacted-langfuse-secret]");
    assert.equal(redacted.self, "[circular-ref]");
  });

  it("does not falsely flag the same object in two branches as circular", () => {
    const shared = { v: "ok" };
    const redact = createSecretRedactor();
    const redacted = redact({ a: shared, b: shared }) as { a: { v: string }; b: { v: string } };
    assert.equal(redacted.a.v, "ok");
    assert.equal(redacted.b.v, "ok");
  });

  it("keeps a literal __proto__ key as data instead of touching the prototype", () => {
    const redact = createSecretRedactor();
    const payload = JSON.parse('{"__proto__": { "x": "sk-lf-hidden" }, "safe": 1 }');
    const redacted = redact(payload) as Record<string, unknown>;
    assert.equal(Object.getPrototypeOf(redacted), Object.prototype);
    assert.deepEqual(Object.getOwnPropertyDescriptor(redacted, "__proto__")?.value, {
      x: "[redacted-langfuse-secret]",
    });
    assert.equal(redacted.safe, 1);
  });

  it("leaves non-strings untouched", () => {
    const redact = createSecretRedactor();
    assert.deepEqual(redact({ n: 42, b: true, x: null }), { n: 42, b: true, x: null });
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

  // Asserting only cost ⊆ usage let the reasoning split ship a usage bucket
  // with no cost bucket, so the mirror must hold in both directions.
  it("mirrors usage and cost keys in both directions, reasoning included", () => {
    const withReasoning: PiUsage = {
      input: 500,
      output: 300,
      cacheRead: 100,
      cacheWrite: 50,
      reasoning: 200,
      cost: { input: 0.001, output: 0.003, cacheRead: 0.00002, cacheWrite: 0.0001875, total: 0.0042075 },
    };
    const usageKeys = Object.keys(buildUsageDetails(withReasoning) ?? {}).sort();
    const costKeys = Object.keys(buildCostDetails(withReasoning) ?? {})
      .filter((k) => k !== "total")
      .sort();
    assert.deepEqual(costKeys, usageKeys);
  });

  it("splits the output cost proportionally and keeps the total exact", () => {
    const details = buildCostDetails({
      input: 500,
      output: 300,
      cacheRead: 0,
      cacheWrite: 0,
      reasoning: 200,
      cost: { input: 0.001, output: 0.003, cacheRead: 0, cacheWrite: 0, total: 0.004 },
    });
    // $0.003 for 300 tokens = $0.00001/token: 100 -> $0.001, 200 -> $0.002.
    assert.ok(Math.abs((details?.output ?? 0) - 0.001) < 1e-12);
    assert.ok(Math.abs((details?.output_reasoning_tokens ?? 0) - 0.002) < 1e-12);
    // Derived by subtraction, so the two buckets must re-sum exactly.
    assert.equal((details?.output ?? 0) + (details?.output_reasoning_tokens ?? 0), 0.003);
  });

  it("implies the same per-token rate for both output buckets", () => {
    const usageWithReasoning: PiUsage = {
      input: 0,
      output: 587,
      cacheRead: 0,
      cacheWrite: 0,
      reasoning: 229,
      cost: { input: 0, output: 0.008805, cacheRead: 0, cacheWrite: 0, total: 0.008805 },
    };
    const u = buildUsageDetails(usageWithReasoning) ?? {};
    const c = buildCostDetails(usageWithReasoning) ?? {};
    const outputRate = (c.output ?? 0) / (u.output ?? 1);
    const reasoningRate = (c.output_reasoning_tokens ?? 0) / (u.output_reasoning_tokens ?? 1);
    assert.ok(Math.abs(outputRate - reasoningRate) < 1e-15, `${outputRate} != ${reasoningRate}`);
    // 15 USD / 1M tokens, the mock rate card.
    assert.ok(Math.abs(outputRate - 15 / 1e6) < 1e-15);
  });

  it("prices the whole output under one key when the reasoning count is inconsistent", () => {
    const details = buildCostDetails({
      input: 0,
      output: 100,
      cacheRead: 0,
      cacheWrite: 0,
      reasoning: 300,
      cost: { input: 0, output: 0.0015, cacheRead: 0, cacheWrite: 0, total: 0.0015 },
    });
    assert.deepEqual(details, { total: 0.0015, output: 0.0015 });
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

describe("extractImages", () => {
  it("returns image parts and ignores everything else", () => {
    const content = [
      { type: "text", text: "look:" },
      { type: "image", data: "aGVsbG8=", mimeType: "image/png" },
      { type: "toolCall", id: "t1", name: "bash" },
      { type: "image", data: 42, mimeType: "image/png" }, // malformed: non-string data
    ];
    assert.deepEqual(extractImages(content), [
      { type: "image", data: "aGVsbG8=", mimeType: "image/png" },
    ]);
  });

  it("returns [] for strings and non-arrays", () => {
    assert.deepEqual(extractImages("just text"), []);
    assert.deepEqual(extractImages(undefined), []);
  });
});

describe("toMultimodalContent", () => {
  it("stays a plain string when there are no images", () => {
    assert.equal(toMultimodalContent("hi", []), "hi");
    assert.equal(toMultimodalContent("hi", undefined), "hi");
  });

  it("emits text + data-URI image parts so the span processor uploads media", () => {
    const parts = toMultimodalContent("what is this?", [
      { type: "image", data: "aGVsbG8=", mimeType: "image/png" },
    ]);
    assert.deepEqual(parts, [
      { type: "text", text: "what is this?" },
      { type: "image_url", image_url: { url: "data:image/png;base64,aGVsbG8=" } },
    ]);
  });

  it("omits the text part when the text is empty (image-only tool results)", () => {
    const parts = toMultimodalContent("", [
      { type: "image", data: "aGVsbG8=", mimeType: "image/webp" },
    ]);
    assert.deepEqual(parts, [
      { type: "image_url", image_url: { url: "data:image/webp;base64,aGVsbG8=" } },
    ]);
  });

  it("drops images the media regex would truncate, keeping the text", () => {
    // URL-safe base64 makes the processor match a prefix and upload garbage.
    assert.equal(toMultimodalContent("hi", [{ type: "image", data: "aGV-bG8_", mimeType: "image/png" }]), "hi");
    // A mime type with parameters breaks the `data:[^;]+;base64,` match.
    assert.equal(
      toMultimodalContent("hi", [{ type: "image", data: "aGVsbG8=", mimeType: "image/svg+xml;charset=utf-8" }]),
      "hi",
    );
    assert.equal(toMultimodalContent("hi", [{ type: "image", data: "", mimeType: "image/png" }]), "hi");
  });

  it("tolerates wrapped base64 by stripping whitespace", () => {
    const parts = toMultimodalContent("", [{ type: "image", data: "aGVs\nbG8=", mimeType: "image/png" }]);
    assert.deepEqual(parts, [
      { type: "image_url", image_url: { url: "data:image/png;base64,aGVsbG8=" } },
    ]);
  });
});

describe("describeImage / renderContentWithImageMarkers", () => {
  const png = { type: "image" as const, data: "a".repeat(4096), mimeType: "image/png" };

  it("reports the decoded size in KB", () => {
    assert.equal(describeImage(png), "[image image/png ~3KB]");
  });

  it("falls back when the mime type is missing", () => {
    assert.equal(describeImage({ type: "image", data: "aGVsbG8=", mimeType: "" }), "[image unknown type ~0KB]");
  });

  it("replaces image parts with markers and keeps the surrounding text", () => {
    assert.equal(
      renderContentWithImageMarkers([
        { type: "text", text: "Read image file [image/png]" },
        png,
      ]),
      "Read image file [image/png]\n[image image/png ~3KB]",
    );
  });

  it("renders an image-only result as a marker rather than nothing", () => {
    // Without this the follow-up generation would read "tool returned nothing".
    assert.equal(renderContentWithImageMarkers([png]), "[image image/png ~3KB]");
  });

  it("never emits base64", () => {
    assert.ok(!renderContentWithImageMarkers([png]).includes(png.data.slice(0, 32)));
  });
});

describe("malformed image parts (untyped tool results)", () => {
  it("describeImage never throws and omits the size for non-string data", () => {
    assert.equal(describeImage({ data: 42, mimeType: "image/png" }), "[image image/png]");
    assert.equal(describeImage({}), "[image unknown type]");
    assert.equal(describeImage({ data: "", mimeType: "image/png" }), "[image image/png]");
  });

  it("renderContentWithImageMarkers survives malformed parts", () => {
    const out = renderContentWithImageMarkers([
      { type: "image" }, // no data at all
      { type: "image", data: 42, mimeType: "image/png" },
      { type: "text", text: 7 }, // non-string text
      { type: "text", text: "ok" },
    ]);
    assert.equal(out, "[image unknown type]\n[image image/png]\nok");
  });
});

describe("readInheritedParent", () => {
  const traceId = "0af7651916cd43dd8448eb211c80319c";
  const spanId = "b7ad6b7169203331";

  it("returns undefined without propagated ids (the normal top-level run)", () => {
    assert.equal(readInheritedParent({}), undefined);
  });

  it("reads a remote parent span context plus the parent session", () => {
    const parent = readInheritedParent({
      LANGFUSE_PI_PARENT_TRACE_ID: traceId,
      LANGFUSE_PI_PARENT_SPAN_ID: spanId,
      LANGFUSE_PI_PARENT_SESSION_ID: "sess-1",
      LANGFUSE_PI_PARENT_DEPTH: "2",
    });
    assert.equal(parent?.spanContext.traceId, traceId);
    assert.equal(parent?.spanContext.spanId, spanId);
    assert.equal(parent?.spanContext.isRemote, true);
    assert.equal(parent?.sessionId, "sess-1");
    assert.equal(parent?.depth, 2);
  });

  it("rejects malformed ids instead of attaching spans to a bogus trace", () => {
    assert.equal(readInheritedParent({ LANGFUSE_PI_PARENT_TRACE_ID: traceId }), undefined);
    assert.equal(readInheritedParent({ LANGFUSE_PI_PARENT_SPAN_ID: spanId }), undefined);
    assert.equal(readInheritedParent({ LANGFUSE_PI_PARENT_TRACE_ID: "nope", LANGFUSE_PI_PARENT_SPAN_ID: spanId }), undefined);
    assert.equal(readInheritedParent({ LANGFUSE_PI_PARENT_TRACE_ID: traceId, LANGFUSE_PI_PARENT_SPAN_ID: "short" }), undefined);
  });

  it("rejects the all-zero ids that OTel defines as invalid", () => {
    assert.equal(
      readInheritedParent({ LANGFUSE_PI_PARENT_TRACE_ID: "0".repeat(32), LANGFUSE_PI_PARENT_SPAN_ID: spanId }),
      undefined,
    );
    assert.equal(
      readInheritedParent({ LANGFUSE_PI_PARENT_TRACE_ID: traceId, LANGFUSE_PI_PARENT_SPAN_ID: "0".repeat(16) }),
      undefined,
    );
  });

  it("defaults depth to 0 when it is missing or junk", () => {
    const base = { LANGFUSE_PI_PARENT_TRACE_ID: traceId, LANGFUSE_PI_PARENT_SPAN_ID: spanId };
    assert.equal(readInheritedParent(base)?.depth, 0);
    assert.equal(readInheritedParent({ ...base, LANGFUSE_PI_PARENT_DEPTH: "abc" })?.depth, 0);
  });
});
