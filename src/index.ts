
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { type ExtensionAPI, type ExtensionContext, getAgentDir } from "@earendil-works/pi-coding-agent";
import { LangfuseSpanProcessor } from "@langfuse/otel";
import {
  type LangfuseSpan,
  type LangfuseGeneration,
  LangfuseOtelSpanAttributes,
  type LangfuseTool,
  setLangfuseTracerProvider,
  startObservation,
} from "@langfuse/tracing";
import { type SpanContext, TraceFlags } from "@opentelemetry/api";
import { AlwaysOnSampler, NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { captureProviderPayload, contextRegistry, maskTelemetry, memoryBlockHashes, type ParentEnvelope } from "./telemetry.ts";
export { captureProviderPayload } from "./telemetry.ts";

const EXTENSION_NAME = "@langfuse/pi-observability-plugin";
const EXTENSION_VERSION = "0.1.2";
const ROOT_OBSERVATION_NAME = "Conversational Turn";
const SUBAGENT_ROOT_OBSERVATION_NAME = "Subagent Turn";
const TRACE_NAME = "Pi Turn";
const GENERATION_PREFIX = "LLM Call";
const TOOL_PREFIX = "Tool:";
const COMPACTION_OBSERVATION_NAME = "Compaction";
const BRANCH_SUMMARY_OBSERVATION_NAME = "Branch Summary";
const TOOL_USAGE_OBSERVATION_NAME = "Tool LLM Usage";
const BASE_TAGS = ["pi"];
const MAX_CHARS = Number(process.env.PI_LANGFUSE_MAX_CHARS ?? "20000");

// A data: URI is only worth emitting if the processor swaps it for a media
// reference; with uploads off it stays in the span as full base64.
const EMIT_IMAGE_MEDIA = (() => {
  const raw = process.env.LANGFUSE_MEDIA_UPLOAD_ENABLED?.trim().toLowerCase();
  return raw ? !["false", "0"].includes(raw) : true;
})();

// pi subagents are child processes that get process.env from the parent. pi
// itself has no trace propagation, so these variables connect the traces.
const ENV_PARENT_TRACE_ID = "LANGFUSE_PI_PARENT_TRACE_ID";
const ENV_PARENT_SPAN_ID = "LANGFUSE_PI_PARENT_SPAN_ID";
const ENV_PARENT_SESSION_ID = "LANGFUSE_PI_PARENT_SESSION_ID";
const ENV_PARENT_DEPTH = "LANGFUSE_PI_PARENT_DEPTH";

const HEX_TRACE_ID = /^[0-9a-f]{32}$/;
const HEX_SPAN_ID = /^[0-9a-f]{16}$/;

export interface InheritedParent {
  spanContext: SpanContext;
  sessionId?: string;
  depth: number;
}

export function readInheritedParent(env: NodeJS.ProcessEnv = process.env): InheritedParent | undefined {
  const traceId = env[ENV_PARENT_TRACE_ID]?.trim().toLowerCase();
  const spanId = env[ENV_PARENT_SPAN_ID]?.trim().toLowerCase();
  if (!traceId || !spanId) return undefined;
  if (!HEX_TRACE_ID.test(traceId) || !HEX_SPAN_ID.test(spanId)) return undefined;
  // OTel defines the all-zero ids as invalid.
  if (/^0+$/.test(traceId) || /^0+$/.test(spanId)) return undefined;
  const depth = Number(env[ENV_PARENT_DEPTH] ?? "0");
  return {
    spanContext: {
      traceId,
      spanId,
      traceFlags: TraceFlags.SAMPLED,
      isRemote: true,
    },
    sessionId: env[ENV_PARENT_SESSION_ID]?.trim() || undefined,
    depth: Number.isFinite(depth) && depth > 0 ? depth : 0,
  };
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

export interface LangfuseConfig {
  publicKey: string;
  secretKey: string;
  baseUrl: string;
  userId?: string;
  environment?: string;
  release?: string;
}

export function loadConfig(): LangfuseConfig | undefined {
  // Kill switch: wins over both env keys and the config file.
  if ((process.env.LANGFUSE_TRACING_ENABLED ?? "").trim().toLowerCase() === "false") {
    return undefined;
  }
  const file = readConfigFile();
  const asTrimmedString = (v: unknown): string | undefined =>
    typeof v === "string" && v.trim() ? v.trim() : undefined;

  const publicKey = process.env.LANGFUSE_PUBLIC_KEY?.trim() || asTrimmedString(file.publicKey);
  const secretKey = process.env.LANGFUSE_SECRET_KEY?.trim() || asTrimmedString(file.secretKey);
  if (!publicKey || !secretKey) return undefined;
  const baseUrl =
    process.env.LANGFUSE_BASE_URL?.trim() ||
    process.env.LANGFUSE_HOST?.trim() ||
    asTrimmedString(file.baseUrl) ||
    "https://cloud.langfuse.com";
  return {
    publicKey,
    secretKey,
    baseUrl: baseUrl.replace(/\/$/, ""),
    userId: process.env.LANGFUSE_USER_ID?.trim() || asTrimmedString(file.userId),
    environment: process.env.LANGFUSE_TRACING_ENVIRONMENT?.trim() || asTrimmedString(file.environment),
    release: process.env.LANGFUSE_RELEASE?.trim() || asTrimmedString(file.release),
  };
}

/**
 * Persistent config lives at `<agentDir>/langfuse.json` (usually
 * `~/.pi/agent/langfuse.json`, keep it chmod 600) so plain `pi` in any project
 * is traced without exporting env vars. Environment variables override the
 * file for ad-hoc runs. Literal values only (no env interpolation, no
 * command execution — a config file must not be able to run code).
 */
function readConfigFile(): Partial<Record<keyof LangfuseConfig, unknown>> {
  try {
    const path = join(getAgentDir(), "langfuse.json");
    if (!existsSync(path)) return {};
    const parsed: unknown = JSON.parse(readFileSync(path, "utf8"));
    return parsed && typeof parsed === "object" ? (parsed as Record<string, unknown>) : {};
  } catch (error) {
    console.error(
      `[pi-langfuse] Ignoring unreadable langfuse.json: ${(error as Error).message}`,
    );
    return {};
  }
}

// ---------------------------------------------------------------------------
// Payload helpers
// ---------------------------------------------------------------------------

export interface TruncationMeta {
  truncated: boolean;
  orig_len: number;
  kept_len?: number;
  sha256?: string;
}

export function truncateText(text: string): { text: string; meta: TruncationMeta } {
  if (text.length <= MAX_CHARS) {
    return { text, meta: { truncated: false, orig_len: text.length } };
  }
  return {
    text: text.slice(0, MAX_CHARS),
    meta: {
      truncated: true,
      orig_len: text.length,
      kept_len: MAX_CHARS,
      sha256: createHash("sha256").update(text).digest("hex"),
    },
  };
}

const SECRET_REDACTION_MARK = "[redacted-langfuse-secret]";
const CYCLE_MARK = "[circular-ref]";
const LANGFUSE_KEY_TOKEN = String.raw`\b[sp]k-lf-[\w-]+\b`;
type TelemetryValue = boolean | number | string | null | undefined | TelemetryValue[] | { [key: string]: TelemetryValue };

function escapeRegExpLiteral(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Redact Langfuse API keys and the given literal secrets from arbitrarily shaped payloads; cycles collapse to a marker so the result survives JSON serialization. */
export function createSecretRedactor(...extraSecrets: string[]): (value: unknown) => TelemetryValue {
  const alternatives = extraSecrets.filter((s) => s.length > 0).map(escapeRegExpLiteral);
  alternatives.push(LANGFUSE_KEY_TOKEN);
  const pattern = new RegExp(alternatives.join("|"), "g");
  const walk = (value: unknown, ancestors: readonly object[]): TelemetryValue => {
    if (typeof value === "string") return value.replace(pattern, SECRET_REDACTION_MARK);
    if (value === null || typeof value === "boolean" || typeof value === "number") return value;
    if (typeof value !== "object") return undefined;
    if (ancestors.includes(value)) return CYCLE_MARK;
    const chain = [...ancestors, value];
    if (Array.isArray(value)) {
      const items: TelemetryValue[] = [];
      for (const item of value) items.push(walk(item, chain));
      return items;
    }
    const fields: { [key: string]: TelemetryValue } = {};
    for (const [key, field] of Object.entries(value)) {
      // defineProperty, not assignment: a key literally named "__proto__"
      // must stay a data key instead of mutating the clone's prototype.
      Object.defineProperty(fields, key, {
        value: walk(field, chain),
        enumerable: true,
        writable: true,
        configurable: true,
      });
    }
    return fields;
  };
  return (value) => walk(value, []);
}

const redactLangfuseKeys = createSecretRedactor();

export function readSystemPrompt(ctx: { getSystemPrompt?: () => string | undefined }): string | undefined {
  try {
    const prompt = ctx.getSystemPrompt?.();
    return typeof prompt === "string" && prompt.trim() ? redactLangfuseKeys(prompt) as string : undefined;
  } catch {
    return undefined;
  }
}

/** Extract plain text from a pi message content array. */
export function extractText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter((p): p is { type: string; text: string } => !!p && typeof p === "object" && (p as { type?: string }).type === "text")
    .map((p) => p.text)
    .join("");
}

export function extractToolCalls(content: unknown): Array<{ id: string; name: string }> {
  if (!Array.isArray(content)) return [];
  return content
    .filter((p): p is { type: string; id: string; name: string } => !!p && typeof p === "object" && (p as { type?: string }).type === "toolCall")
    .map((p) => ({ id: p.id, name: p.name }));
}

export interface PiImagePart {
  type: "image";
  data: string;
  mimeType: string;
}

export function extractImages(content: unknown): PiImagePart[] {
  if (!Array.isArray(content)) return [];
  return content.filter(
    (p): p is PiImagePart =>
      !!p &&
      typeof p === "object" &&
      (p as { type?: string }).type === "image" &&
      typeof (p as { data?: unknown }).data === "string" &&
      typeof (p as { mimeType?: unknown }).mimeType === "string",
  );
}

export function describeImage(image: { type?: unknown; data?: unknown; mimeType?: unknown }): string {
  const mime = typeof image.mimeType === "string" && image.mimeType ? image.mimeType : "unknown type";
  if (typeof image.data !== "string" || !image.data) return `[image ${mime}]`;
  const kb = Math.floor((image.data.length * 3) / 4 / 1024);
  return `[image ${mime} ~${kb}KB]`;
}

export function renderContentWithImageMarkers(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .map((p) => {
      if (!p || typeof p !== "object") return "";
      const part = p as { type?: string; text?: string };
      if (part.type === "text") return typeof part.text === "string" ? part.text : "";
      if (part.type === "image") return describeImage(part as { data?: unknown; mimeType?: unknown });
      return "";
    })
    .filter(Boolean)
    .join("\n");
}

export type ContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string } };

/**
 * pi hands us raw base64. The processor detects media with
 * `/data:[^;]+;base64,[A-Za-z0-9+/]+=*​/`, which silently matches a *prefix* of
 * anything else and uploads that as a corrupt file.
 */
export function toDataUri(image: PiImagePart): string | undefined {
  const data = image.data.replace(/\s+/g, "");
  if (!data || !/^[A-Za-z0-9+/]+={0,2}$/.test(data)) return undefined;
  if (!image.mimeType || image.mimeType.includes(";")) return undefined;
  return `data:${image.mimeType};base64,${data}`;
}

/**
 * Returns `text` unchanged when there are no images. With images it returns the
 * OpenAI-style content parts (the text, then one `image_url` per image) that the
 * Langfuse UI shows as a picture. Never truncate the result: a cut data URI is
 * uploaded as a corrupt file.
 */
export function toMultimodalContent(
  text: string,
  images: readonly PiImagePart[] | undefined,
): string | ContentPart[] {
  const urls = (images ?? []).map(toDataUri).filter((url): url is string => !!url);
  if (!urls.length) return text;
  return [
    ...(text ? [{ type: "text" as const, text }] : []),
    ...urls.map((url) => ({ type: "image_url" as const, image_url: { url } })),
  ];
}

export interface PiUsage {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
  reasoning?: number;
  /** Subset of `cacheWrite`. Only Anthropic reports it. */
  cacheWrite1h?: number;
  cost?: { input: number; output: number; cacheRead: number; cacheWrite: number; total: number };
}

/**
 * Whether the reasoning tokens can be split out of `output`.
 *
 * pi reports `reasoning` as a subset of `output`. A provider that reports more
 * reasoning than output would make the split produce a negative bucket, so both
 * {@link buildUsageDetails} and {@link buildCostDetails} fall back to a single
 * `output` bucket in that case. They MUST agree, otherwise a usage bucket ends
 * up without its cost counterpart.
 */
function resolveReasoningSplit(usage: PiUsage): { reasoning: number; canSplit: boolean } {
  const reasoning = usage.reasoning ?? 0;
  return { reasoning, canSplit: reasoning > 0 && reasoning <= usage.output };
}

/**
 * Changes the pi usage counts into Langfuse usage details. Langfuse counts each
 * token in one key only, but pi includes the reasoning tokens in `output`.
 */
export function buildUsageDetails(usage: PiUsage): Record<string, number> | undefined {
  const details: Record<string, number> = {};
  if (usage.input > 0) details.input = usage.input;
  const { reasoning, canSplit } = resolveReasoningSplit(usage);
  const output = canSplit ? usage.output - reasoning : usage.output;
  if (output > 0) details.output = output;
  if (canSplit) details.output_reasoning_tokens = reasoning;
  if (usage.cacheRead > 0) details.cache_read_input_tokens = usage.cacheRead;
  if (usage.cacheWrite > 0) details.cache_creation_input_tokens = usage.cacheWrite;
  return Object.keys(details).length ? details : undefined;
}

/**
 * Cost keys must mirror the usage keys ({@link buildUsageDetails}) — Langfuse
 * joins the two by name, and server-side pricing emits the same spellings. A
 * usage bucket without its cost twin makes the bucket's implied per-token rate
 * wrong in the UI even though the total stays correct.
 */
export function buildCostDetails(usage: PiUsage): Record<string, number> | undefined {
  const cost = usage.cost;
  if (!cost || !(cost.total > 0)) return undefined;
  const details: Record<string, number> = { total: cost.total };
  if (cost.input > 0) details.input = cost.input;
  if (cost.output > 0) {
    // pi prices every output token of a call at one rate (its tier selection
    // reads only input-side tokens), so the reasoning share is exactly
    // proportional. Deriving the non-reasoning bucket by subtraction keeps the
    // two buckets summing bit-for-bit to the total pi reported.
    const { reasoning, canSplit } = resolveReasoningSplit(usage);
    if (canSplit) {
      const reasoningCost = cost.output * (reasoning / usage.output);
      const nonReasoningCost = cost.output - reasoningCost;
      if (nonReasoningCost > 0) details.output = nonReasoningCost;
      if (reasoningCost > 0) details.output_reasoning_tokens = reasoningCost;
    } else {
      details.output = cost.output;
    }
  }
  if (cost.cacheRead > 0) details.cache_read_input_tokens = cost.cacheRead;
  if (cost.cacheWrite > 0) details.cache_creation_input_tokens = cost.cacheWrite;
  return details;
}

// ---------------------------------------------------------------------------
// Runtime (isolated OTEL provider — never touches the global provider)
// ---------------------------------------------------------------------------

type TraceAttributes = Record<string, string | string[]>;

interface Runtime {
  processor: LangfuseSpanProcessor;
  provider: NodeTracerProvider;
  shutdown: boolean;
}

function isLangfuseSpan(span: { attributes: Record<string, unknown> }): boolean {
  const observationType = span.attributes["langfuse.observation.type"];
  return typeof observationType === "string";
}

function createRuntime(
  config: LangfuseConfig,
  resolveTraceAttributes: () => TraceAttributes,
): Runtime {
  const redactSecrets = createSecretRedactor(config.publicKey, config.secretKey);
  const processor = new LangfuseSpanProcessor({
    publicKey: config.publicKey,
    secretKey: config.secretKey,
    baseUrl: config.baseUrl,
    environment: config.environment,
    release: config.release,
    mask: ({ data }) => maskTelemetry(redactSecrets(data)),
    shouldExportSpan: ({ otelSpan }) => isLangfuseSpan(otelSpan),
  });
  // Trace fields are stamped here rather than on the root span: Langfuse reads
  // them per span, and an extension must not install the global OTel context
  // manager that propagateAttributes() would need.
  const baseOnStart = processor.onStart.bind(processor);
  processor.onStart = (span, parentContext) => {
    baseOnStart(span, parentContext);
    span.setAttributes(resolveTraceAttributes());
  };
  const provider = new NodeTracerProvider({
    spanProcessors: [processor],
    sampler: new AlwaysOnSampler(),
    spanLimits: { attributeValueLengthLimit: Infinity, attributeCountLimit: Infinity },
  });
  setLangfuseTracerProvider(provider);
  return { processor, provider, shutdown: false };
}

// ---------------------------------------------------------------------------
// Per-prompt trace state
// ---------------------------------------------------------------------------

interface OpenGeneration {
  obs: LangfuseGeneration;
  index: number;
  sawFirstToken: boolean;
  finished: boolean;
}

interface PromptState {
  root: LangfuseSpan;
  turnNumber: number;
  generationCount: number;
  openGeneration?: OpenGeneration;
  openTools: Map<string, { obs: LangfuseTool; name: string; startedAt: Date }>;
  pendingToolResults: Array<{ tool_call_id: string; name: string; content: string }>;
  lastAssistantText?: string;
  sawError: boolean;
  userText: string;
  turnImages: PiImagePart[];
  sessionId: string;
  systemPrompt?: string;
  memoryInjection?: Record<string, unknown>;
  memoryRetrievals: Array<{ retrievalId: string; observationId: string }>;
}

const DEBUG = process.env.PI_LANGFUSE_DEBUG === "true";
const debug = (...args: unknown[]) => {
  if (DEBUG) console.error("[pi-langfuse]", ...args);
};

export default function (pi: ExtensionAPI) {
  debug("factory start");
  const config = loadConfig();
  if (!config) {
    pi.on("session_start", (_event, ctx) => {
      if (ctx.hasUI) {
        ctx.ui.setStatus("langfuse", "langfuse: off (no keys)");
      }
    });
    return;
  }

  let runtime: Runtime | undefined;
  let state: PromptState | undefined;
  let traceAttributes: TraceAttributes = {};
  let gitBranch: string | undefined;
  let sessionHadImages = false;
  let fallbackTurnCounter = 0;
  let compactionStartedAt: Date | undefined;
  const environmentParent = readInheritedParent();
  let inheritedParent = environmentParent;
  let explicitParent: ParentEnvelope | undefined;
  let isChildSession = Boolean(environmentParent);
  let lastContext: ExtensionContext | undefined;
  const inheritedParentEnv: Record<string, string | undefined> = {
    [ENV_PARENT_TRACE_ID]: process.env[ENV_PARENT_TRACE_ID],
    [ENV_PARENT_SPAN_ID]: process.env[ENV_PARENT_SPAN_ID],
    [ENV_PARENT_SESSION_ID]: process.env[ENV_PARENT_SESSION_ID],
    [ENV_PARENT_DEPTH]: process.env[ENV_PARENT_DEPTH],
  };

  const ensureRuntime = (): Runtime => {
    if (!runtime || runtime.shutdown) runtime = createRuntime(config, () => traceAttributes);
    // In-process children share the Langfuse SDK's provider pointer.
    setLangfuseTracerProvider(runtime.provider);
    return runtime;
  };

  const readExplicitParent = (): ParentEnvelope | undefined => {
    let parent: ParentEnvelope | undefined;
    pi.events.emit("pi:trace-parent-request", { reply: (value: ParentEnvelope) => { parent = value; } });
    return parent;
  };

  const resolveTurnNumber = (ctx: ExtensionContext, promptText: string): number => {
    try {
      const entries = ctx.sessionManager.getEntries() as Array<{
        type?: string;
        message?: { role?: string; content?: unknown };
      }>;
      const userMessages = entries.filter(
        (e) => e.type === "message" && e.message?.role === "user",
      );
      let count = userMessages.length;
      const last = userMessages[userMessages.length - 1];
      // before_agent_start may fire before or after the prompt is appended.
      if (last && extractText(last.message?.content) !== promptText) count += 1;
      if (count < 1) count = 1;
      fallbackTurnCounter = count;
      return count;
    } catch {
      fallbackTurnCounter += 1;
      return fallbackTurnCounter;
    }
  };

  const closeDanglingObservations = (reason: "interrupted" | "superseded") => {
    if (!state) return;
    for (const [, tool] of state.openTools) {
      tool.obs.update({ level: "WARNING", statusMessage: `Tool run ${reason}`, metadata: { [reason]: true } });
      tool.obs.end();
    }
    state.openTools.clear();
    const gen = state.openGeneration;
    if (gen && !gen.finished) {
      gen.obs.update({ level: "WARNING", statusMessage: `Generation ${reason}`, metadata: { [reason]: true } });
      gen.obs.end();
      gen.finished = true;
    }
    state.openGeneration = undefined;
  };

  // The turn root is the parent for subagents. pi runs tool calls in parallel,
  // so one global variable cannot point to one of many tool spans.
  const publishParentContext = (root: LangfuseSpan, sessionId: string) => {
    const ctx = root.otelSpan.spanContext();
    // Do not publish a root that the sampler dropped. Child spans would point
    // to a trace with no exported root.
    if (!(ctx.traceFlags & TraceFlags.SAMPLED)) return;
    contextRegistry().set(sessionId, Object.freeze({
      traceId: ctx.traceId, spanId: ctx.spanId,
      rootSessionId: inheritedParent?.sessionId ?? sessionId,
      parentSessionId: explicitParent?.parentSessionId,
      depth: inheritedParent?.depth ?? 0,
    }));
    if (isChildSession) {
      const parent = explicitParent?.parentSessionId ? contextRegistry().get(explicitParent.parentSessionId) : undefined;
      if (parent && explicitParent && parent.spanId === explicitParent.spanId && explicitParent.runId && explicitParent.childIndex !== undefined) {
        contextRegistry().set(explicitParent.parentSessionId!, Object.freeze({...parent,
          tracedChildren: [...new Set([...(parent.tracedChildren ?? []), `${explicitParent.runId}:${explicitParent.childIndex}`])],
        }));
      }
      return;
    }
    process.env[ENV_PARENT_TRACE_ID] = ctx.traceId;
    process.env[ENV_PARENT_SPAN_ID] = ctx.spanId;
    process.env[ENV_PARENT_SESSION_ID] = sessionId;
    process.env[ENV_PARENT_DEPTH] = String((inheritedParent?.depth ?? 0) + 1);
  };

  // A child must not attach to a turn that has ended. The inherited values
  // stay valid, so put them back instead of a plain delete.
  const withdrawParentContext = () => {
    if (state && contextRegistry().get(state.sessionId)?.spanId === state.root.otelSpan.spanContext().spanId) {
      contextRegistry().delete(state.sessionId);
    }
    if (isChildSession) return;
    for (const [key, value] of Object.entries(inheritedParentEnv)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  };

  const finalizeRoot = (opts: { cancelled: boolean }) => {
    if (!state) return;
    closeDanglingObservations("interrupted");
    const { text, meta } = truncateText(state.lastAssistantText ?? "");
    const media = EMIT_IMAGE_MEDIA ? state.turnImages : [];
    if (media.length) sessionHadImages = true;
    state.root.update({
      input: media.length ? { role: "user", content: toMultimodalContent(state.userText, media) } : undefined,
      output: state.lastAssistantText ? { role: "assistant", content: text } : undefined,
      level: state.sawError ? "ERROR" : undefined,
      metadata: {
        assistant_text_meta: meta,
        ...(state.turnImages.length ? { image_count: state.turnImages.length } : {}),
        ...(opts.cancelled ? { cancelled: true } : {}),
      },
    });
    state.root.end();
    withdrawParentContext();
    state = undefined;
  };

  // Flushing must never block pi's exit or the next prompt indefinitely.
  // The SDK's forceFlush awaits pending media uploads before the span export,
  // so a session that uploaded images gets a larger budget on the exit path —
  // process.exit would abort an in-flight upload and leave a media token
  // without its binary. Mid-session, timing out is harmless: the process
  // lives on and the upload finishes in the background.
  const FLUSH_TIMEOUT_MS = 3000;
  const EXIT_FLUSH_WITH_MEDIA_TIMEOUT_MS = 15000;
  const flush = async (budgetMs: number = FLUSH_TIMEOUT_MS) => {
    if (!runtime) return;
    try {
      let timer: NodeJS.Timeout | undefined;
      const raced = await Promise.race([
        runtime.processor.forceFlush().then(() => "done" as const),
        new Promise<"timeout">((resolve) => {
          timer = setTimeout(() => resolve("timeout"), budgetMs);
          timer.unref?.();
        }),
      ]);
      if (timer) clearTimeout(timer);
      if (raced === "timeout" && sessionHadImages) {
        console.error(
          `[pi-langfuse] flush timed out after ${budgetMs}ms with media pending — images may be missing from the trace`,
        );
      }
    } catch {
      // Tracing must never break the session.
    }
  };

  pi.on("session_start", async (_event, ctx) => {
    lastContext = ctx;
    debug("session_start");
    if (ctx.hasUI) ctx.ui.setStatus("langfuse", "langfuse ✓");
    try {
      const result = await pi.exec("git", ["rev-parse", "--abbrev-ref", "HEAD"], { timeout: 1000 });
      gitBranch = result.code === 0 ? result.stdout.trim() : undefined;
    } catch {
      gitBranch = undefined;
    }
  });

  const beginPrompt = (event: { prompt: string; images?: unknown }, ctx: ExtensionContext, trigger: string) => {
    lastContext = ctx;
    debug("before_agent_start");
    ensureRuntime();
    debug("runtime ready");
    // A previous prompt that never settled (e.g. rapid re-prompt) is closed
    // rather than leaked.
    if (state) finalizeRoot({ cancelled: true });

    explicitParent = readExplicitParent();
    // Missing explicit IDs mean orphan, not permission to borrow sibling env.
    inheritedParent = explicitParent ? readInheritedParent({
      [ENV_PARENT_TRACE_ID]: explicitParent.traceId,
      [ENV_PARENT_SPAN_ID]: explicitParent.spanId,
      [ENV_PARENT_SESSION_ID]: explicitParent.rootSessionId ?? explicitParent.parentSessionId,
      [ENV_PARENT_DEPTH]: String(explicitParent.depth ?? 1),
    }) : environmentParent;
    isChildSession = Boolean(explicitParent || inheritedParent);

    const sessionId = ctx.sessionManager.getSessionId();
    const turnNumber = resolveTurnNumber(ctx, event.prompt);
    const promptImages = extractImages(event.images);
    const { text: promptText, meta: userMeta } = truncateText(event.prompt);
    const userText = [promptText, ...promptImages.map(describeImage)].filter(Boolean).join("\n");
    const isSubagent = isChildSession;
    traceAttributes = {
      ...(isSubagent ? {} : { [LangfuseOtelSpanAttributes.TRACE_NAME]: TRACE_NAME }),
      [LangfuseOtelSpanAttributes.TRACE_SESSION_ID]: isSubagent
        ? (inheritedParent?.sessionId ?? explicitParent?.rootSessionId ?? sessionId)
        : sessionId,
      [LangfuseOtelSpanAttributes.TRACE_TAGS]: BASE_TAGS,
      ...(config.userId ? { [LangfuseOtelSpanAttributes.TRACE_USER_ID]: config.userId } : {}),
    };

    const root = startObservation(
      isSubagent ? SUBAGENT_ROOT_OBSERVATION_NAME : ROOT_OBSERVATION_NAME,
      {
        input: { role: "user", content: userText },
        metadata: {
          source: "pi",
          extension: EXTENSION_NAME,
          extension_version: EXTENSION_VERSION,
          session_id: sessionId,
          turn_number: turnNumber,
          trigger,
          parent_link_source: explicitParent ? (inheritedParent ? "explicit" : "unavailable") : inheritedParent ? "environment" : "none",
          cwd: ctx.cwd,
          user_text_meta: userMeta,
          ...(gitBranch ? { git_branch: gitBranch } : {}),
          ...(ctx.model ? { model: ctx.model.id, provider: ctx.model.provider } : {}),
          ...(isSubagent
            ? { pi_subagent: true, subagent_depth: explicitParent?.depth ?? inheritedParent?.depth,
                parent_session_id: explicitParent?.parentSessionId ?? inheritedParent?.sessionId,
                root_session_id: inheritedParent?.sessionId ?? explicitParent?.rootSessionId,
                parent_trace_id: inheritedParent?.spanContext.traceId,
                parent_observation_id: inheritedParent?.spanContext.spanId,
                run_id: explicitParent?.runId, agent: explicitParent?.agent, child_index: explicitParent?.childIndex,
                parent_tool_call_id: explicitParent?.parentToolCallId, source_run_id: explicitParent?.sourceRunId }
            : {}),
        },
      },
      { asType: "span", ...(inheritedParent ? { parentSpanContext: inheritedParent.spanContext } : {}) },
    );

    state = {
      root,
      turnNumber,
      generationCount: 0,
      openTools: new Map(),
      pendingToolResults: [],
      sawError: false,
      userText,
      turnImages: [...promptImages],
      sessionId,
      memoryRetrievals: [],
    };
    publishParentContext(root, sessionId);
    debug("root created, turn", turnNumber);
  };

  pi.on("before_agent_start", (event, ctx) => beginPrompt(event, ctx, "before_agent_start"));
  pi.on("agent_start", (_event, ctx) => {
    lastContext = ctx;
    if (!state) {
      const latest = [...ctx.sessionManager.getBranch()].reverse().find(entry => entry.type === "message");
      beginPrompt({ prompt: latest?.type === "message" ? extractText((latest.message as {content?: unknown}).content) : "" }, ctx, "agent_start");
    }
    const systemPrompt = readSystemPrompt(ctx);
    if (!state || !systemPrompt) return;
    state.systemPrompt = systemPrompt;
    try { state.root.update({ metadata: { system_prompt: systemPrompt } }); } catch { /* tracing is best effort */ }
  });

  pi.events.on("hindsight:retrieval", (raw: unknown) => {
    // Telemetry is optional: observer failures must not alter recall behavior.
    try {
      if (!raw || typeof raw !== "object" || !lastContext) return;
      const event = raw as Record<string, unknown>;
      if (event.version !== 1 || !["retrieval", "injection"].includes(String(event.phase))) return;
      if (event.sessionId && event.sessionId !== lastContext.sessionManager.getSessionId()) return;
      const standalone = !state;
      if (!state) beginPrompt({prompt:""}, lastContext, "memory-event");
      if (!state) return;
      ensureRuntime();
      const { results, query, ...details } = event;
      const date = new Date(String(event.startedAt));
      const filters = event.filters && typeof event.filters === "object" ? event.filters as Record<string, unknown> : {};
      const attributes = {
          input: event.phase === "retrieval" ? {query, bank_id:event.bankId, tag_groups:event.tagGroups, filters:event.filters, budget:event.budget ?? filters.budget, max_tokens:event.maxTokens ?? filters.maxTokens} : undefined,
          output: event.phase === "retrieval" ? {results, kept_ids:event.keptIds, injected_ids:event.injectedIds} : {injected:event.injected, rendered_hash:event.renderedHash, rendered_length:event.renderedLength},
          level: event.status === "error" || event.status === "timeout" ? "WARNING" as const : "DEFAULT" as const,
          statusMessage: typeof event.error === "string" ? event.error : undefined,
          metadata: {...details, bank_id:event.bankId, retrieval_id:event.retrievalId, context_id:event.contextId},
      };
      const timing = Number.isFinite(date.getTime()) ? {startTime:date} : {};
      const observation = event.phase === "retrieval"
        ? state.root.startObservation("Hindsight Recall", attributes, {asType:"retriever", ...timing})
        : state.root.startObservation("Hindsight Context Injection", attributes, {asType:"event", ...timing});
      observation.end();
      if (event.phase === "retrieval" && typeof event.retrievalId === "string") state.memoryRetrievals.push({retrievalId:event.retrievalId, observationId:observation.otelSpan.spanContext().spanId});
      if (event.phase === "injection") state.memoryInjection = event;
      if (standalone) { finalizeRoot({cancelled:false}); void flush(); }
    } catch { /* Tracing must never break the session. */ }
  });

  pi.on("before_provider_request", (event, ctx) => {
    if (!state) beginPrompt({ prompt: "" }, ctx, "provider-fallback");
    if (!state) return;
    ensureRuntime();
    // A new provider request while one is open means the previous HTTP
    // attempt was retried/superseded — close it instead of leaking it.
    if (state.openGeneration && !state.openGeneration.finished) {
      const gen = state.openGeneration;
      gen.obs.update({ level: "WARNING", statusMessage: "Superseded by provider retry", metadata: { superseded: true } });
      gen.obs.end();
    }
    const index = ++state.generationCount;
    const assembled = captureProviderPayload(event.payload);
    const memoryHashes = memoryBlockHashes(event.payload);

    const obs = state.root.startObservation(
      GENERATION_PREFIX,
      {
        input: assembled.input,
        model: ctx.model?.id,
        metadata: {
          assistant_index: index - 1,
          assembled_context: assembled.meta,
          memory_block_hashes: memoryHashes,
          memory_context_id: state.memoryInjection?.contextId,
          memory_retrieval_ids: state.memoryInjection?.retrievalIds,
          memory_retrieval_observation_ids: state.memoryRetrievals.filter(item =>
            (state?.memoryInjection?.retrievalIds as string[] | undefined)?.includes(item.retrievalId)).map(item => item.observationId),
          memory_injection_verified: state.memoryInjection?.injected === true
            ? memoryHashes.includes(String(state.memoryInjection.renderedHash)) : undefined,
          ...(ctx.model ? { provider: ctx.model.provider } : {}),
        },
      },
      { asType: "generation" },
    );
    state.openGeneration = { obs, index, sawFirstToken: false, finished: false };
  });

  pi.on("message_update", (event) => {
    const gen = state?.openGeneration;
    if (!gen || gen.finished || gen.sawFirstToken) return;
    if (extractText((event.message as { content?: unknown })?.content).length > 0) {
      gen.sawFirstToken = true;
      gen.obs.update({ completionStartTime: new Date() });
    }
  });

  pi.on("message_end", (event) => {
    if (!state) return;
    const message = event.message as {
      role: string;
      content?: unknown;
      model?: string;
      responseModel?: string;
      provider?: string;
      api?: string;
      responseId?: string;
      stopReason?: string;
      errorMessage?: string;
      usage?: PiUsage;
    };
    if (message.role !== "assistant") return;
    const gen = state.openGeneration;
    if (!gen || gen.finished) return;

    const text = extractText(message.content);
    const tools = extractToolCalls(message.content);
    const { text: outText, meta: outMeta } = truncateText(text);
    const isError = message.stopReason === "error" || message.stopReason === "aborted";
    if (message.stopReason === "error") state.sawError = true;

    gen.obs.update({
      output: {
        role: "assistant",
        ...(outText ? { content: outText } : {}),
        ...(tools.length ? { tool_calls: tools } : {}),
      },
      model: message.responseModel || message.model,
      usageDetails: message.usage ? buildUsageDetails(message.usage) : undefined,
      costDetails: message.usage ? buildCostDetails(message.usage) : undefined,
      ...(isError
        ? {
            level: "ERROR" as const,
            statusMessage: message.errorMessage || `stopReason: ${message.stopReason}`,
          }
        : {}),
      metadata: {
        assistant_text_meta: outMeta,
        tool_count: tools.length,
        ...(message.stopReason ? { stop_reason: message.stopReason } : {}),
        ...(message.responseId ? { response_id: message.responseId } : {}),
        ...(message.api ? { api: message.api } : {}),
        ...(message.responseModel && message.responseModel !== message.model
          ? { requested_model: message.model }
          : {}),
        ...(message.usage?.cacheWrite1h ? { cache_write_1h_tokens: message.usage.cacheWrite1h } : {}),
      },
    });
    gen.obs.end();
    gen.finished = true;
    state.openGeneration = undefined;
    if (text) state.lastAssistantText = text;
    state.pendingToolResults = [];
  });

  pi.on("tool_execution_start", (event) => {
    if (!state) return;
    ensureRuntime();
    const redactedArgs = redactLangfuseKeys(event.args) as Record<string, unknown>;
    const serializedArgs = safeStringify(redactedArgs);
    const hasDataUri = /data:[^;,]{0,100};base64,/.test(serializedArgs);
    let input: unknown = redactedArgs;
    let argsMeta: TruncationMeta | undefined;
    if (hasDataUri || serializedArgs.length > MAX_CHARS) {
      const marked = serializedArgs.replace(
        /data:[^;,]{0,100};base64,[A-Za-z0-9+/]+=*/g,
        (uri) => `[data uri ~${Math.floor((uri.length * 3) / 4 / 1024)}KB]`,
      );
      const t = truncateText(marked);
      input = t.text;
      argsMeta = t.meta;
    }
    const obs = state.root.startObservation(
      `${TOOL_PREFIX} ${event.toolName}`,
      {
        input,
        metadata: {
          tool_name: event.toolName,
          tool_id: event.toolCallId,
          ...(argsMeta ? { args_meta: argsMeta } : {}),
        },
      },
      { asType: "tool" },
    );
    state.openTools.set(event.toolCallId, { obs, name: event.toolName, startedAt: new Date() });
  });

  pi.on("tool_execution_end", (event, ctx) => {
    if (!state) return;
    const open = state.openTools.get(event.toolCallId);
    if (!open) return;
    state.openTools.delete(event.toolCallId);

    const result = event.result as { content?: unknown; usage?: PiUsage; details?: {runId?: string; results?: Array<{index?: number}>} } | undefined;
    const images = extractImages(result?.content);
    const rawOutput = renderContentWithImageMarkers(result?.content) || safeStringify(result?.content);
    const { text: outText, meta: outMeta } = truncateText(rawOutput);
    if (event.isError) state.sawError = true;
    state.turnImages.push(...images);

    const childResults = result?.details?.results;
    const tracedChildren = contextRegistry().get(state.sessionId)?.tracedChildren ?? [];
    const childUsageAlreadyTraced = event.toolName === "subagent" && Boolean(childResults?.length)
      && childResults!.every(child => child.index !== undefined && tracedChildren.includes(`${result?.details?.runId}:${child.index}`));
    if (childUsageAlreadyTraced) open.obs.update({metadata:{usage_source:"linked_child_generations", aggregate_usage_omitted:true}});
    if (result?.usage && !childUsageAlreadyTraced) {
      ensureRuntime();
      const usageObs = startObservation(
        TOOL_USAGE_OBSERVATION_NAME,
        {
          usageDetails: buildUsageDetails(result.usage),
          costDetails: buildCostDetails(result.usage),
          // Best effort: pi does not record which model the tool used
          // internally. The session model keeps the no-cost fallback priceable;
          // the flag below marks the attribution as approximate.
          model: ctx.model?.id,
          metadata: {
            tool_call_id: event.toolCallId,
            tool_name: open.name,
            source: "tool_result_usage",
            model_is_session_model: true,
            ...(result.usage.cacheWrite1h ? { cache_write_1h_tokens: result.usage.cacheWrite1h } : {}),
          },
        },
        {
          asType: "generation",
          parentSpanContext: open.obs.otelSpan.spanContext(),
          startTime: open.startedAt,
        },
      );
      usageObs.end();
      debug("tool result usage traced", open.name, result.usage.input, result.usage.output);
    }

    open.obs.update({
      output: outText || undefined,
      ...(event.isError ? { level: "ERROR" as const, statusMessage: "Tool execution failed" } : {}),
      metadata: {
        output_meta: outMeta,
        is_error: Boolean(event.isError),
        ...(images.length ? { image_count: images.length } : {}),
      },
    });
    open.obs.end();

    state.pendingToolResults.push({
      tool_call_id: event.toolCallId,
      name: open.name,
      content: outText.slice(0, 4000),
    });
  });

  const emitSummarizationGeneration = (
    name: string,
    ctx: ExtensionContext,
    attributes: {
      output?: { role: string; content: string };
      model?: string;
      usageDetails?: Record<string, number>;
      costDetails?: Record<string, number>;
      metadata: Record<string, unknown>;
    },
    startedAt?: Date,
  ): void => {
    ensureRuntime();
    if (state) {
      const obs = startObservation(name, attributes, {
        asType: "generation",
        parentSpanContext: state.root.otelSpan.spanContext(),
        ...(startedAt ? { startTime: startedAt } : {}),
      });
      obs.end();
      return;
    }
    // ensureRuntime is otherwise only called in before_agent_start; an idle
    // /compact in a fresh process would hit the unset tracer provider.
    ensureRuntime();
    const sessionId = ctx.sessionManager.getSessionId();
    traceAttributes = {
      ...(inheritedParent ? {} : { [LangfuseOtelSpanAttributes.TRACE_NAME]: `Pi ${name}` }),
      [LangfuseOtelSpanAttributes.TRACE_SESSION_ID]: inheritedParent?.sessionId ?? sessionId,
      [LangfuseOtelSpanAttributes.TRACE_TAGS]: BASE_TAGS,
      ...(config.userId ? { [LangfuseOtelSpanAttributes.TRACE_USER_ID]: config.userId } : {}),
    };
    const obs = startObservation(
      name,
      {
        ...attributes,
        metadata: {
          ...attributes.metadata,
          source: "pi",
          extension: EXTENSION_NAME,
          extension_version: EXTENSION_VERSION,
          session_id: sessionId,
        },
      },
      {
        asType: "generation",
        ...(startedAt ? { startTime: startedAt } : {}),
        ...(inheritedParent ? { parentSpanContext: inheritedParent.spanContext } : {}),
      },
    );
    obs.end();
    void flush();
  }

  pi.on("session_before_compact", () => {
    compactionStartedAt = new Date();
  });

  pi.on("session_compact", (event, ctx) => {
    const startedAt = compactionStartedAt;
    compactionStartedAt = undefined;
    const entry = event.compactionEntry as
      | { summary?: string; tokensBefore?: number; usage?: PiUsage; fromHook?: boolean }
      | undefined;
    const usage = entry?.usage;
    const { text: summaryText, meta: summaryMeta } = truncateText(entry?.summary ?? "");
    emitSummarizationGeneration(
      COMPACTION_OBSERVATION_NAME,
      ctx,
      {
        output: summaryText ? { role: "assistant", content: summaryText } : undefined,
        model: ctx.model?.id,
        usageDetails: usage ? buildUsageDetails(usage) : undefined,
        costDetails: usage ? buildCostDetails(usage) : undefined,
        metadata: {
          compaction_reason: event.reason,
          will_retry: event.willRetry,
          from_extension: event.fromExtension,
          ...(usage?.cacheWrite1h ? { cache_write_1h_tokens: usage.cacheWrite1h } : {}),
          ...(entry?.fromHook ? { from_hook: true } : {}),
          ...(typeof entry?.tokensBefore === "number" ? { tokens_before: entry.tokensBefore } : {}),
          ...(ctx.model ? { provider: ctx.model.provider } : {}),
          summary_meta: summaryMeta,
        },
      },
      startedAt,
    );
    debug("compaction traced", usage?.input, usage?.output, usage?.cost?.total);
  });

  // Branch summaries (session tree navigation) are the second slice of pi's
  // "Tools/summaries" bucket: a real summarization call whose usage lands on
  // the BranchSummaryEntry that session_tree exposes.
  pi.on("session_tree", (event, ctx) => {
    const entry = event.summaryEntry as
      | { summary?: string; usage?: PiUsage; fromHook?: boolean }
      | undefined;
    if (!entry) return; // plain navigation without a summarization call
    const usage = entry.usage;
    const { text: summaryText, meta: summaryMeta } = truncateText(entry.summary ?? "");
    emitSummarizationGeneration(
      BRANCH_SUMMARY_OBSERVATION_NAME,
      ctx,
      {
        output: summaryText ? { role: "assistant", content: summaryText } : undefined,
        model: ctx.model?.id,
        usageDetails: usage ? buildUsageDetails(usage) : undefined,
        costDetails: usage ? buildCostDetails(usage) : undefined,
        metadata: {
          ...(event.fromExtension ? { from_extension: true } : {}),
          ...(entry.fromHook ? { from_hook: true } : {}),
          ...(usage?.cacheWrite1h ? { cache_write_1h_tokens: usage.cacheWrite1h } : {}),
          ...(ctx.model ? { provider: ctx.model.provider } : {}),
          summary_meta: summaryMeta,
        },
      },
    );
    debug("branch summary traced", usage?.input, usage?.output, usage?.cost?.total);
  });

  pi.on("agent_settled", async (_event, ctx) => {
    if (!state) return;
    finalizeRoot({ cancelled: false });
    await flush();
    if (ctx.hasUI) ctx.ui.setStatus("langfuse", "langfuse ✓ (trace sent)");
  });

  pi.on("session_shutdown", async (event) => {
    if (state) finalizeRoot({ cancelled: true });
    const exitBudget = sessionHadImages ? EXIT_FLUSH_WITH_MEDIA_TIMEOUT_MS : FLUSH_TIMEOUT_MS;
    await flush(exitBudget);
    if (event.reason === "quit" && runtime && !runtime.shutdown) {
      runtime.shutdown = true;
      try {
        let timer: NodeJS.Timeout | undefined;
        await Promise.race([
          runtime.provider.shutdown(),
          new Promise<void>((resolve) => {
            timer = setTimeout(resolve, exitBudget);
            timer.unref?.();
          }),
        ]);
        if (timer) clearTimeout(timer);
      } catch {
        // ignore
      }
    }
  });
}

function safeStringify(value: unknown): string {
  if (value === undefined || value === null) return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}
