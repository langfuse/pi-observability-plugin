/**
 * Test extension that rewrites the system prompt for the turn. Loaded after
 * the Langfuse extension, so it is the "later-registered extension" from
 * issue #21: an observer that reads the prompt during before_agent_start
 * would still see the original, only agent_start sees this override.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  pi.on("before_agent_start", (_event, ctx) => {
    const suffix = process.env.TEST_SYSTEM_PROMPT_SUFFIX ?? "";
    return { systemPrompt: `${ctx.getSystemPrompt()}\n\n${suffix}` };
  });
}
