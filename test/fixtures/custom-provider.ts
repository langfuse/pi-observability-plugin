/**
 * Test provider whose streamSimple never calls options.onPayload. pi therefore
 * never emits before_provider_request for it, which mirrors bridge-style
 * providers such as pi-claude-bridge.
 */
import {
  type AssistantMessage,
  type AssistantMessageEventStream,
  type Context,
  type Model,
  type SimpleStreamOptions,
  createAssistantMessageEventStream,
} from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export const CUSTOM_PROVIDER = "custom-no-payload";
export const CUSTOM_MODEL = "custom-model";
export const CUSTOM_REPLY = "Reply from a custom provider.";

export const CUSTOM_USAGE = {
  input: 120,
  output: 8,
  cacheRead: 40,
  cacheWrite: 10,
  totalTokens: 178,
  cost: { input: 0.001, output: 0.002, cacheRead: 0.0001, cacheWrite: 0.0002, total: 0.0033 },
};

function streamWithoutPayload(
  model: Model<string>,
  _context: Context,
  _options?: SimpleStreamOptions,
): AssistantMessageEventStream {
  const stream = createAssistantMessageEventStream();

  (async () => {
    const output: AssistantMessage = {
      role: "assistant",
      content: [],
      api: model.api,
      provider: model.provider,
      model: model.id,
      usage: CUSTOM_USAGE,
      stopReason: "pending",
      timestamp: Date.now(),
    };

    stream.push({ type: "start", partial: output });
    output.content.push({ type: "text", text: "" });
    stream.push({ type: "text_start", contentIndex: 0, partial: output });

    const block = output.content[0];
    if (block?.type === "text") block.text = CUSTOM_REPLY;
    stream.push({ type: "text_delta", contentIndex: 0, delta: CUSTOM_REPLY, partial: output });
    stream.push({ type: "text_end", contentIndex: 0, content: CUSTOM_REPLY, partial: output });

    output.stopReason = "stop";
    stream.push({ type: "done", reason: "stop", message: output });
    stream.end();
  })();

  return stream;
}

export default function (pi: ExtensionAPI) {
  pi.registerProvider(CUSTOM_PROVIDER, {
    name: "Custom Provider Without onPayload",
    baseUrl: "http://127.0.0.1:1",
    apiKey: "unused",
    api: CUSTOM_PROVIDER,
    streamSimple: streamWithoutPayload,
    models: [
      {
        id: CUSTOM_MODEL,
        name: "Custom Model",
        reasoning: false,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 128000,
        maxTokens: 8192,
      },
    ],
  });
}
