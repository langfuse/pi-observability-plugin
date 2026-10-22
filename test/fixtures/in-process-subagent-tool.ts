/**
 * Test subagent tool that runs the child agent in the same process, the way
 * orchestration extensions do: a second AgentSession built with the pi SDK,
 * with this extension loaded again, and disposed through
 * AgentSessionRuntime.dispose(), which emits session_shutdown "quit".
 *
 * TEST_SUBAGENT_OVERLAP=1 starts a second child while the first child's turn
 * is still open, so the second child must not attach to the first one.
 */
import {
  type CreateAgentSessionRuntimeFactory,
  type ExtensionAPI,
  type ExtensionContext,
  createAgentSessionFromServices,
  createAgentSessionRuntime,
  createAgentSessionServices,
  getAgentDir,
  SessionManager,
} from "@earendil-works/pi-coding-agent";

async function startChild(ctx: ExtensionContext, task: string) {
  const parentSession = ctx.sessionManager.getSessionId();
  const createRuntime: CreateAgentSessionRuntimeFactory = async ({ cwd, agentDir, sessionManager, sessionStartEvent }) => {
    const services = await createAgentSessionServices({
      cwd,
      agentDir,
      resourceLoaderOptions: {
        additionalExtensionPaths: [process.env.TEST_LANGFUSE_EXTENSION!],
        noContextFiles: true,
      },
    });
    return {
      ...(await createAgentSessionFromServices({ services, sessionManager, sessionStartEvent, model: ctx.model })),
      services,
      diagnostics: services.diagnostics,
    };
  };
  const runtime = await createAgentSessionRuntime(createRuntime, {
    cwd: ctx.cwd,
    agentDir: getAgentDir(),
    sessionManager: SessionManager.inMemory(ctx.cwd, { parentSession }),
  });
  await runtime.session.bindExtensions({});
  const started = new Promise<void>((resolve) => {
    const unsubscribe = runtime.session.subscribe((event: { type?: string }) => {
      if (event.type === "agent_start") {
        unsubscribe();
        resolve();
      }
    });
  });
  const done = runtime.session.prompt(`Task: ${task}`);
  return { runtime, started, done };
}

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
    async execute(
      _toolCallId: string,
      params: { task: string },
      _signal: AbortSignal | undefined,
      _onUpdate: unknown,
      ctx: ExtensionContext,
    ) {
      const children = [await startChild(ctx, params.task)];
      if (process.env.TEST_SUBAGENT_OVERLAP === "1") {
        await children[0]!.started;
        children.push(await startChild(ctx, `${params.task} (second)`));
      }
      await Promise.all(children.map((child) => child.done));
      for (const child of children) await child.runtime.dispose();
      return { content: [{ type: "text", text: `${children.length} subagent(s) finished.` }], details: {} };
    },
  } as never);
}
