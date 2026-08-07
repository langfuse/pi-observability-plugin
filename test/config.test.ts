import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
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

  beforeEach(() => {
    saved = Object.fromEntries(CONFIG_VARS.map((k) => [k, process.env[k]]));
    // Point at a fresh, empty agent dir so the developer's real
    // ~/.pi/agent/langfuse.json can never leak into tests.
    agentDir = mkdtempSync(join(tmpdir(), "pi-lf-cfg-"));
    for (const k of CONFIG_VARS) delete process.env[k];
    process.env.PI_CODING_AGENT_DIR = agentDir;
  });

  afterEach(() => {
    for (const k of CONFIG_VARS) {
      if (saved[k] === undefined) delete process.env[k];
      else process.env[k] = saved[k];
    }
  });

  const writeConfigFile = (data: unknown) =>
    writeFileSync(join(agentDir, "langfuse.json"), JSON.stringify(data));

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
