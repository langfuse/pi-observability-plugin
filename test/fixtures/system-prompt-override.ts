import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  pi.on("before_agent_start", (_event, ctx) => {
    const suffix = process.env.TEST_SYSTEM_PROMPT_SUFFIX ?? "";
    return { systemPrompt: `${ctx.getSystemPrompt()}\n\n${suffix}` };
  });
}
