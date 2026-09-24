/**
 * Test fixture. It prints the parent-context variable at two points, so a test
 * can check that the extension publishes it in the turn and withdraws it after.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  pi.on("before_provider_request", async () => {
    console.error(`PROBE turn ${process.env.LANGFUSE_PI_PARENT_TRACE_ID ?? "<unset>"}`);
    console.error(`PROBE turn-tp ${process.env.LANGFUSE_PI_TRACEPARENT ?? "<unset>"}`);
    console.error(`PROBE turn-ext ${process.env.LANGFUSE_PI_PARENT_EXTERNAL_TRACE ?? "<unset>"}`);
  });
  pi.on("agent_settled", async () => {
    console.error(`PROBE settled ${process.env.LANGFUSE_PI_PARENT_TRACE_ID ?? "<unset>"}`);
    console.error(`PROBE settled-tp ${process.env.LANGFUSE_PI_TRACEPARENT ?? "<unset>"}`);
  });
}
