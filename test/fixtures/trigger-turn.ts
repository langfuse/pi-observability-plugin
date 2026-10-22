/**
 * Test fixture. After the first prompt settles, it starts a second agent run
 * with sendMessage({ triggerTurn: true }), as pi's file-trigger example does.
 * pi starts that run without before_agent_start. Print mode does not wait for
 * the run, so the handler waits until pi is idle again.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export const TRIGGER_TEXT = "External trigger: summarize the workspace";

export default function (pi: ExtensionAPI) {
  let fired = false;
  pi.on("agent_settled", async (_event, ctx) => {
    if (fired) return;
    fired = true;
    pi.sendMessage({ customType: "file-trigger", content: TRIGGER_TEXT, display: true }, { triggerTurn: true });
    const deadline = Date.now() + 30_000;
    while (!ctx.isIdle() && Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 50));
    }
  });
}
