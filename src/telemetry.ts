import { createHash } from "node:crypto";

const REDACTED = "[redacted-secret]";
const SECRET_FIELD = /^(?:authorization|proxy-authorization|cookie|set-cookie|api[-_]?key|secret[-_]?key|secret|password|passwd|client[-_]?secret|(?:api|auth|access|refresh|id)[-_]?token|token|credentials|connection[-_]?string|database[-_]?url|private[-_]?key|encrypted_content)$/i;
type SanitizedTelemetry = boolean | number | string | null | undefined | SanitizedTelemetry[] | { [key: string]: SanitizedTelemetry };

/** Copy only; never mutate the provider request or memory result. */
export function sanitizeTelemetry(value: unknown, secrets: string[] = [], ancestors: object[] = [], omitBinary = true): SanitizedTelemetry {
  if (typeof value === "string") {
    let text = value;
    for (const secret of secrets) if (secret.length >= 8) text = text.split(secret).join(REDACTED);
    if (omitBinary) text = text.replace(/data:[^;,\s]{1,100};base64,[A-Za-z0-9+/\s]+=*/g, "[binary data omitted]");
    return text
      .replace(/\b(Bearer|Basic)\s+[A-Za-z0-9._~+\/-]+=*/gi, `$1 ${REDACTED}`)
      .replace(/\b[sp]k-lf-[\w-]+\b/g, "[redacted-langfuse-secret]")
      .replace(/\b(?:sk-(?:proj-|ant-)?[\w-]{8,}|pk-lf-[\w-]+|gh[pousr]_[\w]{16,})\b/g, REDACTED)
      .replace(/([?&](?:api[_-]?key|token|access_token|signature|x-amz-signature)=)[^&\s"'<>]+/gi, `$1${REDACTED}`)
      .replace(/(\b[a-z][a-z\d+.-]*:\/\/)[^/\s:@]+:[^@\s/]+@/gi, `$1[redacted-credentials]@`)
      .replace(/(\b[A-Z_]*(?:API[_-]?KEY|SECRET|PASSWORD|ACCESS_TOKEN|REFRESH_TOKEN)["']?\s*[:=]\s*["']?)[^\s"'`,;}]+/gi, `$1${REDACTED}`);
  }
  if (value === null || typeof value === "boolean" || typeof value === "number") return value;
  if (typeof value !== "object") return undefined;
  if (ancestors.includes(value)) return "[circular-ref]";
  if (ancestors.length >= 80) return "[depth limit]";
  const next = [...ancestors, value];
  if (Array.isArray(value)) return value.map(item => sanitizeTelemetry(item, secrets, next, omitBinary));
  // SAFETY: value is non-null object; only own enumerable properties are copied below.
  const record = value as Record<string, unknown>;
  if (omitBinary && record.type === "image" && (record.data || record.source)) {
    return { type:"image", mimeType:sanitizeTelemetry(record.mimeType, secrets, next, omitBinary), omitted:true };
  }
  const fields: { [key: string]: SanitizedTelemetry } = {};
  for (const [key, item] of Object.entries(record)) {
    const inlineBinary = omitBinary && (key === "file_data" || (key === "data" &&
      (record.mimeType || record.mime_type || record.media_type || record.format || record.type === "base64")));
    fields[key] = SECRET_FIELD.test(key) ? REDACTED : inlineBinary ? "[binary omitted]" : sanitizeTelemetry(item, secrets, next, omitBinary);
  }
  return fields;
}

export function environmentSecrets(): string[] {
  return Object.entries(process.env)
    .filter(([key, value]) => /(?:KEY|SECRET|PASSWORD|TOKEN)$/.test(key) && (value?.length ?? 0) >= 8)
    .map(([, value]) => value!);
}

export function maskTelemetry(data: unknown): SanitizedTelemetry {
  const secrets = environmentSecrets();
  if (typeof data === "string") {
    try { return JSON.stringify(sanitizeTelemetry(JSON.parse(data), secrets, [], false)); }
    catch { return sanitizeTelemetry(data, secrets, [], false); }
  }
  return sanitizeTelemetry(data, secrets, [], false);
}

export function captureProviderPayload(payload: unknown, limit = Number(process.env.PI_LANGFUSE_CONTEXT_MAX_CHARS ?? 1000000)) {
  const request = sanitizeTelemetry(payload, environmentSecrets());
  const serialized = JSON.stringify(request) ?? "null";
  const cap = Number.isFinite(limit) && limit >= 1000 ? Math.floor(limit) : 1000000;
  const truncated = serialized.length > cap;
  const captured = truncated
    ? { truncated:true, head:serialized.slice(0, Math.floor(cap / 2)), tail:serialized.slice(-Math.floor(cap / 2)) }
    : request;
  const input = !truncated && captured && typeof captured === "object" && !Array.isArray(captured)
    && Array.isArray((captured as { messages?: unknown }).messages)
    ? (captured as { messages: unknown[] }).messages
    : captured;
  return {
    input,
    meta: {
      capture_source:"before_provider_request", redacted:true, binary_omitted:true,
      truncated, original_characters:serialized.length, captured_characters:Math.min(serialized.length, cap),
      sha256:createHash("sha256").update(serialized).digest("hex"), request:captured,
    },
  };
}

export interface ParentEnvelope {
  traceId?: string;
  spanId?: string;
  rootSessionId?: string;
  parentSessionId?: string;
  depth?: number;
  runId?: string;
  agent?: string;
  childIndex?: number;
  parentToolCallId?: string;
  sourceRunId?: string;
  tracedChildren?: readonly string[];
}
export const CONTEXT_REGISTRY = Symbol.for("pi.langfuse.contexts.v1");
export function contextRegistry(): Map<string, ParentEnvelope> {
  // SAFETY: this process-local symbol is owned exclusively by this extension integration.
  const global = globalThis as unknown as Record<symbol, unknown>;
  if (!(global[CONTEXT_REGISTRY] instanceof Map)) global[CONTEXT_REGISTRY] = new Map();
  return global[CONTEXT_REGISTRY] as Map<string, ParentEnvelope>;
}

/** Hash the actual memory blocks present in a request, without persisting raw text. */
export function memoryBlockHashes(payload: unknown): string[] {
  const hashes = new Set<string>();
  const seen = new Set<object>();
  const walk = (value: unknown) => {
    if (typeof value === "string") {
      if (value.includes("<hindsight-")) hashes.add(createHash("sha256").update(value).digest("hex"));
      for (const match of value.matchAll(/(?:<hindsight-mental-models>[\s\S]*?<\/hindsight-mental-models>\s*)?<hindsight-memory>[\s\S]*?<\/hindsight-memory>|<hindsight-mental-models>[\s\S]*?<\/hindsight-mental-models>/g)) {
        hashes.add(createHash("sha256").update(match[0]).digest("hex"));
      }
    } else if (value && typeof value === "object" && !seen.has(value)) {
      seen.add(value); for (const item of Object.values(value)) walk(item);
    }
  };
  walk(payload); return [...hashes];
}
