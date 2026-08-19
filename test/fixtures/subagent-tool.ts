/**
 * Test subagent tool. It starts a child pi process that gets process.env from
 * the parent, the same as the pi subagent example and the gallery packages.
 */
import { spawn } from "node:child_process";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "subagent",
    label: "Subagent",
    description: "Delegate a task to an isolated child agent.",
    parameters: {
      type: "object",
      properties: { task: { type: "string", description: "Task for the subagent" } },
      required: ["task"],
    } as never,
    async execute(_toolCallId: string, params: { task: string }) {
      const piBin = process.env.TEST_PI_BIN!;
      const childArgs = [
        "-e", process.env.TEST_LANGFUSE_EXTENSION!,
        "--provider", "mock",
        "--model", "mock-gpt-1",
        "--api-key", "mock-key",
        "--no-context-files",
        "--mode", "json",
        "-p", "--no-session",
        `Task: ${params.task}`,
      ];
      const stdout = await new Promise<string>((resolve) => {
        const child = spawn(piBin, childArgs, { cwd: process.cwd(), shell: false, stdio: ["ignore", "pipe", "pipe"] });
        let buf = "";
        child.stdout.on("data", (d: Buffer) => (buf += d.toString()));
        child.on("close", () => resolve(buf));
      });

      const texts: string[] = [];
      for (const line of stdout.split("\n")) {
        if (!line.trim()) continue;
        try {
          const event = JSON.parse(line) as {
            message?: { role?: string; content?: Array<{ type?: string; text?: string }> };
          };
          if (event?.message?.role === "assistant") {
            for (const part of event.message.content ?? []) {
              if (part?.type === "text" && part.text) texts.push(part.text);
            }
          }
        } catch {
          // pi json mode also prints lines that are not events.
        }
      }
      return { content: [{ type: "text", text: texts.join("\n") || "(subagent produced no text)" }], details: {} };
    },
  } as never);
}
