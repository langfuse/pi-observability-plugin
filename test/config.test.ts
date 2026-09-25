import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, it } from "node:test";
import { loadConfig } from "../src/index.ts";

const CONFIG_VARS = [
  "LANGFUSE_PUBLIC_KEY",
  "LANGFUSE_SECRET_KEY",
  "LANGFUSE_BASE_URL",
  "LANGFUSE_HOST",
  "LANGFUSE_USER_ID",
  "LANGFUSE_TRACING_ENVIRONMENT",
  "LANGFUSE_RELEASE",
  "LANGFUSE_TRACING_ENABLED",
  "PI_CODING_AGENT_DIR",
];

describe("loadConfig", () => {
  let saved: Record<string, string | undefined>;
  let agentDir: string;
  let workspace: string;
  let originalCwd: string;

  beforeEach(() => {
    saved = Object.fromEntries(CONFIG_VARS.map((k) => [k, process.env[k]]));
    // Point at a fresh, empty agent dir so the developer's real
    // ~/.pi/agent/langfuse.json can never leak into tests.
    agentDir = mkdtempSync(join(tmpdir(), "pi-lf-cfg-"));
    originalCwd = process.cwd();
    workspace = mkdtempSync(join(tmpdir(), "pi-lf-cfg-ws-"));
    process.chdir(workspace);
    for (const k of CONFIG_VARS) delete process.env[k];
    process.env.PI_CODING_AGENT_DIR = agentDir;
  });

  afterEach(() => {
    process.chdir(originalCwd);
    rmSync(workspace, { recursive: true, force: true });
    rmSync(agentDir, { recursive: true, force: true });
    for (const k of CONFIG_VARS) {
      if (saved[k] === undefined) delete process.env[k];
      else process.env[k] = saved[k];
    }
  });

  const writeConfigFile = (data: unknown) =>
    writeFileSync(join(agentDir, "langfuse.json"), JSON.stringify(data));

  const writeProjectConfig = (data: unknown) => {
    mkdirSync(join(workspace, ".pi"), { recursive: true });
    writeFileSync(join(workspace, ".pi", "langfuse.json"), JSON.stringify(data));
  };

  it("reads project config without a global config", () => {
    writeProjectConfig({ publicKey: "pk-lf-project", secretKey: "sk-lf-project", baseUrl: "https://project.example.com/" });
    assert.equal(loadConfig()?.publicKey, "pk-lf-project");
    assert.equal(loadConfig()?.secretKey, "sk-lf-project");
    assert.equal(loadConfig()?.baseUrl, "https://project.example.com");
  });

  it("selects project config without merging global values", () => {
    writeConfigFile({ publicKey: "pk-lf-global", secretKey: "sk-lf-global", baseUrl: "https://global.example.com", userId: "global-user" });
    writeProjectConfig({ publicKey: "pk-lf-project", secretKey: "sk-lf-project" });
    assert.deepEqual(loadConfig(), {
      publicKey: "pk-lf-project", secretKey: "sk-lf-project",
      baseUrl: "https://cloud.langfuse.com", userId: undefined,
      environment: undefined, release: undefined,
    });
  });

  it("lets environment variables override project config", () => {
    writeProjectConfig({ publicKey: "pk-lf-project", secretKey: "sk-lf-project", baseUrl: "https://project.example.com", userId: "project-user" });
    process.env.LANGFUSE_PUBLIC_KEY = "pk-lf-env";
    process.env.LANGFUSE_HOST = "https://env.example.com";
    process.env.LANGFUSE_USER_ID = "env-user";
    const config = loadConfig();
    assert.equal(config?.publicKey, "pk-lf-env");
    assert.equal(config?.secretKey, "sk-lf-project");
    assert.equal(config?.baseUrl, "https://env.example.com");
    assert.equal(config?.userId, "env-user");
    process.env.LANGFUSE_TRACING_ENABLED = "false";
    assert.equal(loadConfig(), undefined);
  });

  for (const data of [{}, { publicKey: "pk-lf-project" }]) {
    it(`does not fall back to global keys for incomplete project config ${JSON.stringify(data)}`, () => {
      writeConfigFile({ publicKey: "pk-lf-global", secretKey: "sk-lf-global" });
      writeProjectConfig(data);
      assert.equal(loadConfig(), undefined);
    });
  }

  for (const data of [null, [], "invalid", 42, true]) {
    it(`falls back to global config for non-object project JSON ${JSON.stringify(data)}`, () => {
      writeConfigFile({ publicKey: "pk-lf-global", secretKey: "sk-lf-global" });
      writeProjectConfig(data);
      assert.equal(loadConfig()?.publicKey, "pk-lf-global");
    });
  }

  it("falls back to global config when project JSON is malformed", () => {
    writeConfigFile({ publicKey: "pk-lf-global", secretKey: "sk-lf-global" });
    writeProjectConfig({});
    writeFileSync(join(workspace, ".pi", "langfuse.json"), "{not json");
    assert.equal(loadConfig()?.publicKey, "pk-lf-global");
  });

  it("falls back to global config when the project file cannot be read", () => {
    writeConfigFile({ publicKey: "pk-lf-global", secretKey: "sk-lf-global" });
    mkdirSync(join(workspace, ".pi", "langfuse.json"), { recursive: true });
    assert.equal(loadConfig()?.publicKey, "pk-lf-global");
  });

  it("returns undefined without keys", () => {
    assert.equal(loadConfig(), undefined);
  });

  it("reads keys from env vars", () => {
    process.env.LANGFUSE_PUBLIC_KEY = "pk-lf-env";
    process.env.LANGFUSE_SECRET_KEY = "sk-lf-env";
    const config = loadConfig();
    assert.equal(config?.publicKey, "pk-lf-env");
    assert.equal(config?.baseUrl, "https://cloud.langfuse.com");
  });

  it("reads keys from the config file", () => {
    writeConfigFile({ publicKey: "pk-lf-file", secretKey: "sk-lf-file", baseUrl: "https://eu.example.com/", userId: "u1" });
    const config = loadConfig();
    assert.equal(config?.publicKey, "pk-lf-file");
    assert.equal(config?.baseUrl, "https://eu.example.com"); // trailing slash stripped
    assert.equal(config?.userId, "u1");
  });

  it("lets env vars override the config file", () => {
    writeConfigFile({ publicKey: "pk-lf-file", secretKey: "sk-lf-file", userId: "file-user" });
    process.env.LANGFUSE_PUBLIC_KEY = "pk-lf-env";
    process.env.LANGFUSE_USER_ID = "env-user";
    const config = loadConfig();
    assert.equal(config?.publicKey, "pk-lf-env");
    assert.equal(config?.secretKey, "sk-lf-file"); // not overridden -> file value
    assert.equal(config?.userId, "env-user");
  });

  it("supports LANGFUSE_HOST as baseUrl alias", () => {
    process.env.LANGFUSE_PUBLIC_KEY = "pk-lf-x";
    process.env.LANGFUSE_SECRET_KEY = "sk-lf-x";
    process.env.LANGFUSE_HOST = "https://host.example.com";
    assert.equal(loadConfig()?.baseUrl, "https://host.example.com");
  });

  it("kill switch wins over env keys and config file", () => {
    writeConfigFile({ publicKey: "pk-lf-file", secretKey: "sk-lf-file" });
    process.env.LANGFUSE_PUBLIC_KEY = "pk-lf-env";
    process.env.LANGFUSE_SECRET_KEY = "sk-lf-env";
    process.env.LANGFUSE_TRACING_ENABLED = "false";
    assert.equal(loadConfig(), undefined);
  });

  it("ignores a malformed config file instead of crashing", () => {
    writeFileSync(join(agentDir, "langfuse.json"), "{not json");
    assert.equal(loadConfig(), undefined);
    process.env.LANGFUSE_PUBLIC_KEY = "pk-lf-env";
    process.env.LANGFUSE_SECRET_KEY = "sk-lf-env";
    assert.equal(loadConfig()?.publicKey, "pk-lf-env");
  });

  it("ignores non-string values in the config file", () => {
    writeConfigFile({ publicKey: "pk-lf-file", secretKey: "sk-lf-file", userId: 42, environment: ["dev"] });
    const config = loadConfig();
    assert.equal(config?.userId, undefined);
    assert.equal(config?.environment, undefined);
  });
});
