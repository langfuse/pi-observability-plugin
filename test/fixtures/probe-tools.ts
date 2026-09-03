/**
 * Test extension that registers two tools and then deactivates one, so a
 * trace can show the difference between "registered" (getAllTools) and
 * "offered to the model" (getActiveTools). Neither tool is ever called: the
 * mock provider only drives bash and read.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export const PROBE_ACTIVE_TOOL = {
  name: "probe_active",
  description: "Active probe tool. Stays in the request.",
  parameters: {
    type: "object",
    properties: {
      target: { type: "string", description: "What to probe" },
      depth: { type: "integer", minimum: 0 },
    },
    required: ["target"],
  },
};

export const PROBE_INACTIVE_TOOL = {
  name: "probe_inactive",
  description: "Inactive probe tool. Registered but switched off.",
  parameters: { type: "object", properties: {} },
};

export default function (pi: ExtensionAPI) {
  for (const tool of [PROBE_ACTIVE_TOOL, PROBE_INACTIVE_TOOL]) {
    pi.registerTool({
      name: tool.name,
      label: tool.name,
      description: tool.description,
      parameters: tool.parameters as never,
      async execute() {
        return { content: [{ type: "text", text: "probe" }], details: {} };
      },
    });
  }
  pi.on("session_start", () => {
    pi.setActiveTools(pi.getActiveTools().filter((name) => name !== PROBE_INACTIVE_TOOL.name));
  });
}
