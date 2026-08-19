
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
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";

const EXTENSION_NAME = "@langfuse/pi-observability-plugin";
const EXTENSION_VERSION = "0.0.1";
const ROOT_OBSERVATION_NAME = "Conversational Turn";
const GENERATION_PREFIX = "LLM Call"; 
const TOOL_PREFIX = "Tool:";
const BASE_TAGS = ["pi"];
const MAX_CHARS = Number(process.env.PI_LANGFUSE_MAX_CHARS ?? "20000");

// A data: URI is only worth emitting if the processor swaps it for a media
// reference; with uploads off it stays in the span as full base64.
const EMIT_IMAGE_MEDIA = (() => {
  const raw = process.env.LANGFUSE_MEDIA_UPLOAD_ENABLED?.trim().toLowerCase();
  return raw ? !["false", "0"].includes(raw) : true;
})();

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

export function maskSecrets(data: unknown, secrets: readonly string[]): unknown {
  return maskValue(data, secrets, new WeakSet<object>());
}

function maskValue(data: unknown, secrets: readonly string[], seen: WeakSet<object>): unknown {
  if (typeof data === "string") {
    let masked = data.replace(/\b(?:sk|pk)-lf-[A-Za-z0-9_-]+\b/g, "[LANGFUSE_KEY_REDACTED]");
    for (const secret of secrets) {
      if (secret) masked = masked.replaceAll(secret, "[LANGFUSE_KEY_REDACTED]");
    }
    return masked;
  }
  if (!data || typeof data !== "object") return data;
  if (seen.has(data)) return "[circular]";
  seen.add(data);
  try {
    if (Array.isArray(data)) return data.map((item) => maskValue(item, secrets, seen));
    return Object.fromEntries(
      Object.entries(data).map(([k, v]) => [k, maskValue(v, secrets, seen)]),
    );
  } finally {
    seen.delete(data);
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

export function shortSessionLabel(sessionId: string): string {
  if (!sessionId) return "unknown";
  const parts = sessionId.split("-");
  if (parts.length === 5 && parts[0]?.length === 8) return parts[0];
  return sessionId.slice(0, 12).replace(/-+$/, "") || "unknown";
}

export interface PiUsage {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
  reasoning?: number;
  cost?: { input: number; output: number; cacheRead: number; cacheWrite: number; total: number };
}

/**
 * Changes the pi usage counts into Langfuse usage details. Langfuse counts each
 * token in one key only, but pi includes the reasoning tokens in `output`.
 */
export function buildUsageDetails(usage: PiUsage): Record<string, number> | undefined {
  const details: Record<string, number> = {};
  if (usage.input > 0) details.input = usage.input;
  // Do not split when a provider reports more reasoning than output tokens.
  // A count that is too large makes the total wrong.
  const reasoning = usage.reasoning ?? 0;
  const canSplitReasoning = reasoning > 0 && reasoning <= usage.output;
  const output = canSplitReasoning ? usage.output - reasoning : usage.output;
  if (output > 0) details.output = output;
  if (canSplitReasoning) details.output_reasoning_tokens = reasoning;
  if (usage.cacheRead > 0) details.cache_read_input_tokens = usage.cacheRead;
  if (usage.cacheWrite > 0) details.cache_creation_input_tokens = usage.cacheWrite;
  return Object.keys(details).length ? details : undefined;
}

/**
 * Cost keys must mirror the usage keys ({@link buildUsageDetails}) — Langfuse
 * joins the two by name, and server-side pricing emits the same spellings.
 */
export function buildCostDetails(usage: PiUsage): Record<string, number> | undefined {
  const cost = usage.cost;
  if (!cost || !(cost.total > 0)) return undefined;
  const details: Record<string, number> = { total: cost.total };
  if (cost.input > 0) details.input = cost.input;
  if (cost.output > 0) details.output = cost.output;
  if (cost.cacheRead > 0) details.cache_read_input_tokens = cost.cacheRead;
  if (cost.cacheWrite > 0) details.cache_creation_input_tokens = cost.cacheWrite;
  return details;
}

// ---------------------------------------------------------------------------
// Runtime (isolated OTEL provider — never touches the global provider)
// ---------------------------------------------------------------------------

interface Runtime {
  processor: LangfuseSpanProcessor;
  provider: NodeTracerProvider;
  shutdown: boolean;
}

function createRuntime(config: LangfuseConfig): Runtime {
  const secrets = [config.secretKey, config.publicKey];
  const processor = new LangfuseSpanProcessor({
    publicKey: config.publicKey,
    secretKey: config.secretKey,
    baseUrl: config.baseUrl,
    environment: config.environment,
    release: config.release,
    mask: ({ data }) => maskSecrets(data, secrets),
    shouldExportSpan: ({ otelSpan }) =>
      typeof otelSpan.attributes["langfuse.observation.type"] === "string",
  });
  const provider = new NodeTracerProvider({
    spanProcessors: [processor],
    // OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT etc. must not apply here: a cut data
    // URI still matches the media regex and uploads garbage. Payload size is
    // bounded by our own truncateText instead.
    spanLimits: { attributeValueLengthLimit: Infinity },
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
  openTools: Map<string, { obs: LangfuseTool; name: string }>;
  pendingToolResults: Array<{ tool_call_id: string; name: string; content: string }>;
  lastAssistantText?: string;
  sawError: boolean;
  userText: string;
  turnImages: PiImagePart[];
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
  let gitBranch: string | undefined;
  let sessionHadImages = false;
  let fallbackTurnCounter = 0;
  let lastPromptText = "";

  const ensureRuntime = (): Runtime => {
    if (!runtime || runtime.shutdown) runtime = createRuntime(config);
    return runtime;
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
    debug("session_start");
    if (ctx.hasUI) ctx.ui.setStatus("langfuse", "langfuse ✓");
    try {
      const result = await pi.exec("git", ["rev-parse", "--abbrev-ref", "HEAD"], { timeout: 1000 });
      gitBranch = result.code === 0 ? result.stdout.trim() : undefined;
    } catch {
      gitBranch = undefined;
    }
  });

  pi.on("before_agent_start", (event, ctx) => {
    debug("before_agent_start");
    ensureRuntime();
    debug("runtime ready");
    // A previous prompt that never settled (e.g. rapid re-prompt) is closed
    // rather than leaked.
    if (state) finalizeRoot({ cancelled: true });

    const sessionId = ctx.sessionManager.getSessionId();
    const turnNumber = resolveTurnNumber(ctx, event.prompt);
    const promptImages = extractImages(event.images);
    const { text: promptText, meta: userMeta } = truncateText(event.prompt);
    const userText = [promptText, ...promptImages.map(describeImage)].filter(Boolean).join("\n");
    lastPromptText = userText;

    const root = startObservation(
      ROOT_OBSERVATION_NAME,
      {
        input: { role: "user", content: userText },
        metadata: {
          source: "pi",
          extension: EXTENSION_NAME,
          extension_version: EXTENSION_VERSION,
          session_id: sessionId,
          turn_number: turnNumber,
          cwd: ctx.cwd,
          user_text_meta: userMeta,
          ...(gitBranch ? { git_branch: gitBranch } : {}),
          ...(ctx.model ? { model: ctx.model.id, provider: ctx.model.provider } : {}),
        },
      },
      { asType: "span" },
    );
    // Trace-level fields live as attributes on the root span (the SDK's
    // context-based propagation needs a global context manager, which an
    // extension must not install).
    root.otelSpan.setAttribute(
      LangfuseOtelSpanAttributes.TRACE_NAME,
      `Pi - Turn ${turnNumber} (${shortSessionLabel(sessionId)})`,
    );
    root.otelSpan.setAttribute(LangfuseOtelSpanAttributes.TRACE_SESSION_ID, sessionId);
    root.otelSpan.setAttribute(LangfuseOtelSpanAttributes.TRACE_TAGS, BASE_TAGS);
    if (config.userId) {
      root.otelSpan.setAttribute(LangfuseOtelSpanAttributes.TRACE_USER_ID, config.userId);
    }

    state = {
      root,
      turnNumber,
      generationCount: 0,
      openTools: new Map(),
      pendingToolResults: [],
      sawError: false,
      userText,
      turnImages: [...promptImages],
    };
    debug("root created, turn", turnNumber);
  });

  pi.on("before_provider_request", (_event, ctx) => {
    if (!state) return;
    // A new provider request while one is open means the previous HTTP
    // attempt was retried/superseded — close it instead of leaking it.
    if (state.openGeneration && !state.openGeneration.finished) {
      const gen = state.openGeneration;
      gen.obs.update({ level: "WARNING", statusMessage: "Superseded by provider retry", metadata: { superseded: true } });
      gen.obs.end();
    }
    const index = ++state.generationCount;
    const generationInput =
      index === 1
        ? { role: "user", content: lastPromptText }
        : state.pendingToolResults.length
          ? { role: "tool", tool_results: state.pendingToolResults }
          : undefined;

    const obs = state.root.startObservation(
      GENERATION_PREFIX,
      {
        input: generationInput,
        model: ctx.model?.id,
        metadata: {
          assistant_index: index - 1,
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
      model: message.model,
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
    const maskedArgs = maskSecrets(event.args, []) as Record<string, unknown>;
    const serializedArgs = safeStringify(maskedArgs);
    const hasDataUri = /data:[^;,]{0,100};base64,/.test(serializedArgs);
    let input: unknown = maskedArgs;
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
    state.openTools.set(event.toolCallId, { obs, name: event.toolName });
  });

  pi.on("tool_execution_end", (event) => {
    if (!state) return;
    const open = state.openTools.get(event.toolCallId);
    if (!open) return;
    state.openTools.delete(event.toolCallId);

    const result = event.result as { content?: unknown } | undefined;
    const images = extractImages(result?.content);
    const rawOutput = renderContentWithImageMarkers(result?.content) || safeStringify(result?.content);
    const { text: outText, meta: outMeta } = truncateText(rawOutput);
    if (event.isError) state.sawError = true;
    state.turnImages.push(...images);

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
