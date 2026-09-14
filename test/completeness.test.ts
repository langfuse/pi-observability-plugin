import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { createHash } from "node:crypto";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";
import extension, * as plugin from "../src/index.ts";
import { maskTelemetry, memoryBlockHashes } from "../src/telemetry.ts";
import { startCaptureServer, waitForRequests } from "./helpers.ts";

function harness(sessionId: string, parent?: Record<string, unknown>) {
  const handlers = new Map<string, Function[]>(); const bus = new EventEmitter();
  if (parent) bus.on("pi:trace-parent-request", ({reply}) => reply(parent));
  const ctx: any = {cwd: tmpdir(), hasUI:false, model:{id:"fixture",provider:"fixture"}, sessionManager:{getSessionId:()=>sessionId,getEntries:()=>[],getBranch:()=>[]}};
  const pi: any = {on:(name:string, fn:Function)=>handlers.set(name,[...(handlers.get(name)??[]),fn]), events:{on:(n:string,f:any)=>{bus.on(n,f);return()=>bus.off(n,f);},emit:(n:string,x:any)=>bus.emit(n,x)},exec:async()=>({code:0,stdout:"test\n"})};
  extension(pi);
  return {ctx,bus,emit:async(name:string,event:any={})=>{for(const fn of handlers.get(name)??[])await fn({type:name,...event},ctx);}};
}
async function withCapture(fn: (capture:any)=>Promise<void>) {
  const capture=await startCaptureServer(); const saved={...process.env};
  process.env.PI_CODING_AGENT_DIR=mkdtempSync(join(tmpdir(),"pi-tracing-config-"));
  Object.assign(process.env,{LANGFUSE_PUBLIC_KEY:"pk-lf-test",LANGFUSE_SECRET_KEY:"sk-lf-test",LANGFUSE_BASE_URL:`http://127.0.0.1:${capture.port}`,LANGFUSE_TRACING_ENABLED:"true"});
  for(const key of Object.keys(process.env))if(key.startsWith("LANGFUSE_PI_PARENT_"))delete process.env[key];
  try{await fn(capture);}finally{for(const key of Object.keys(process.env))if(!(key in saved))delete process.env[key];Object.assign(process.env,saved);capture.close();}
}
const metadata=(s:any,k:string)=>s.attrs[`langfuse.observation.metadata.${k}`];

describe("tracing completeness regressions",()=>{
  it("omits provider-native inline image, audio and document bytes",()=>{
    const payload={contents:[{parts:[{inlineData:{mimeType:"image/png",data:"GEMINI_BINARY_SENTINEL"}}]}],audio:{type:"input_audio",input_audio:{data:"AUDIO_BINARY_SENTINEL",format:"wav"}},doc:{type:"document",source:{type:"base64",media_type:"application/pdf",data:"DOCUMENT_BINARY_SENTINEL"}},file:{type:"input_file",file_data:"FILE_BINARY_SENTINEL"}};
    const captured=JSON.stringify((plugin as any).captureProviderPayload(payload));
    for(const value of ["GEMINI_BINARY_SENTINEL","AUDIO_BINARY_SENTINEL","DOCUMENT_BINARY_SENTINEL","FILE_BINARY_SENTINEL"])assert.ok(!captured.includes(value));
  });
  it("verifies combined mental-model and recalled-memory injections",()=>{
    const mental="<hindsight-mental-models>model</hindsight-mental-models>";
    const memory="<hindsight-memory>fact</hindsight-memory>";
    for(const text of [mental,memory,`${mental}\n\n${memory}`]){
      const digest=createHash("sha256").update(text).digest("hex");
      assert.ok(memoryBlockHashes({messages:[{content:text}]}).includes(digest));
      assert.ok(memoryBlockHashes({messages:[{content:`prefix\n${text}\nsuffix`}]}).includes(digest));
    }
  });
  it("preserves upstream opt-in image uploads outside assembled-request capture",()=>{
    const root={role:"user",content:[{type:"image_url",image_url:{url:"data:image/png;base64,AAAA"}}]};
    assert.ok(JSON.stringify(maskTelemetry(root)).includes("data:image/png;base64,AAAA"));
    assert.ok(!JSON.stringify((plugin as any).captureProviderPayload(root)).includes("data:image/png;base64,AAAA"));
  });
  it("redacts credentials embedded in prose and URLs, not only JSON fields",()=>{
    const result=(plugin as any).captureProviderPayload({messages:[{role:"user",content:'API_KEY=fixture-plain-secret\nAuthorization: Basic Zml4dHVyZTpzZWNyZXQ=\npostgresql://user:fixture-db-password@localhost/db'}],token:"fixture-token-value"});
    for(const secret of ['fixture-plain-secret','Zml4dHVyZTpzZWNyZXQ=','fixture-db-password','fixture-token-value'])assert.ok(!JSON.stringify(result).includes(secret),`leaked ${secret}`);
  });
  it("does not double-count a linked native child's usage as a second model call",async()=>withCapture(async capture=>{
    const parent=harness("usage-parent");await parent.emit("before_agent_start",{prompt:"parent"});
    await parent.emit("tool_execution_start",{toolCallId:"t1",toolName:"subagent",args:{agent:"scout"}});
    const registry=(globalThis as any)[Symbol.for("pi.langfuse.contexts.v1")];
    const child=harness("usage-child",{...registry.get("usage-parent"),parentSessionId:"usage-parent",depth:1,runId:"usage-run",agent:"scout",childIndex:0});
    await child.emit("before_agent_start",{prompt:"child"});await child.emit("before_provider_request",{payload:{messages:[]}});
    const usage={input:10,output:5,cacheRead:0,cacheWrite:0,cost:{input:0.01,output:0.01,total:0.02}};
    await child.emit("message_end",{message:{role:"assistant",content:[{type:"text",text:"done"}],usage}});await child.emit("agent_settled");
    await parent.emit("tool_execution_end",{toolCallId:"t1",toolName:"subagent",result:{content:[],usage,details:{runId:"usage-run",results:[{index:0}]}}});await parent.emit("agent_settled");
    await waitForRequests(capture,2);assert.equal(capture.spans().filter((s:any)=>s.name==="Tool LLM Usage").length,0);
    for(const h of [child,parent])await h.emit("session_shutdown",{reason:"quit"});
  }));
  it("captures assembled payload without mutation, preserving history/system/tools and masking secrets",()=>{
    const payload={model:"test",messages:[{role:"system",content:"system instructions"},{role:"user",content:"earlier user"},{role:"assistant",content:"earlier reply"},{role:"user",content:"new turn"}],tools:[{name:"read",parameters:{type:"object"}}],credentials:{api_key:"fixture-private-value"},header:"Bearer fixture-bearer-value",binary:{type:"image",data:"AAAA",mimeType:"image/png"}};
    const before=structuredClone(payload);const capture=(plugin as any).captureProviderPayload;
    assert.equal(typeof capture,"function","provider payload capture must exist");
    const result=capture(payload);assert.deepEqual(payload,before);
    assert.equal(result.input.length,4);assert.deepEqual(result.meta.request.tools,payload.tools);
    assert.ok(!JSON.stringify(result).includes("fixture-private-value"));assert.ok(!JSON.stringify(result).includes("fixture-bearer-value"));assert.ok(!JSON.stringify(result).includes('"data":"AAAA"'));
    assert.equal(result.meta.truncated,false);
  });
  it("captures payload tail beyond old 20K cap and labels intentional limits",()=>{
    const capture=(plugin as any).captureProviderPayload;assert.equal(typeof capture,"function");
    const result=capture({messages:[{role:"system",content:"a".repeat(30000)+"TAIL_SENTINEL"}]});
    assert.ok(JSON.stringify(result.input).includes("TAIL_SENTINEL"));assert.equal(result.meta.truncated,false);
    assert.equal(capture({content:"x".repeat(1500)},1000).meta.truncated,true);
  });
  it("traces autonomous agent_start without before_agent_start and includes actual payload",async()=>withCapture(async capture=>{
    const h=harness("autonomous-session");await h.emit("agent_start");
    const payload={model:"fixture",messages:[{role:"system",content:"ASSEMBLED_SYSTEM"},{role:"user",content:"AUTONOMOUS_RESULT"}],tools:[]};
    await h.emit("before_provider_request",{payload});await h.emit("message_end",{message:{role:"assistant",content:[{type:"text",text:"Done"}]}});await h.emit("agent_settled");
    await waitForRequests(capture,1);const spans=capture.spans();assert.equal(spans.filter((s:any)=>s.name==="LLM Call").length,1);
    const gen=spans.find((s:any)=>s.name==="LLM Call");assert.ok(String(gen.attrs["langfuse.observation.input"]).includes("ASSEMBLED_SYSTEM"));
    assert.equal(metadata(spans.find((s:any)=>s.name==="Conversational Turn"),"trigger"),"agent_start");
    await h.emit("session_shutdown",{reason:"quit"});
  }));
  it("keeps in-process siblings under the true parent and preserves parent attribution after children",async()=>withCapture(async capture=>{
    const parent=harness("parent-session");await parent.emit("before_agent_start",{prompt:"parent",images:[]});
    const registry=(globalThis as any)[Symbol.for("pi.langfuse.contexts.v1")];assert.ok(registry instanceof Map,"session-scoped parent registry missing");
    const envelope={...registry.get("parent-session"),parentSessionId:"parent-session",depth:1,runId:"run-1",agent:"reviewer"};
    const originalSpan=process.env.LANGFUSE_PI_PARENT_SPAN_ID;
    const a=harness("child-a",{...envelope,childIndex:0});await a.emit("before_agent_start",{prompt:"child-a"});
    const b=harness("child-b",{...envelope,childIndex:1});await b.emit("before_agent_start",{prompt:"child-b"});
    assert.equal(process.env.LANGFUSE_PI_PARENT_SPAN_ID,originalSpan,"children must not overwrite parent env");
    for(const h of [b,a,parent]){await h.emit("before_provider_request",{payload:{messages:[{role:"user",content:"test"}]}});await h.emit("message_end",{message:{role:"assistant",content:[{type:"text",text:"done"}]}});await h.emit("agent_settled");}
    await waitForRequests(capture,3);const spans=capture.spans();const root=spans.find((s:any)=>s.name==="Conversational Turn");const children=spans.filter((s:any)=>s.name==="Subagent Turn");assert.equal(children.length,2);
    for(const c of children){assert.equal(c.parentSpanId,root.spanId);assert.equal(c.traceId,root.traceId);assert.equal(metadata(c,"run_id"),"run-1");assert.equal(metadata(c,"agent"),"reviewer");assert.equal(c.attrs["session.id"],"parent-session");}
    assert.ok(spans.filter((s:any)=>s.name==="LLM Call").every((s:any)=>s.attrs["session.id"]==="parent-session"));
    for(const h of [a,b,parent])await h.emit("session_shutdown",{reason:"quit"});
  }));
  it("exports correlated RETRIEVER events and injection linkage on the following generation",async()=>withCapture(async capture=>{
    const h=harness("memory-session");await h.emit("before_agent_start",{prompt:"recall decision"});
    const now=new Date().toISOString();h.bus.emit("hindsight:retrieval",{version:1,phase:"retrieval",mode:"automatic",sessionId:"memory-session",retrievalId:"r1",contextId:"c1",bankId:"fixture-bank",query:"exact scoped query",tagGroups:[{all:[{tags:["project:fixture"],match:"any_strict"}]}],startedAt:now,endedAt:now,durationMs:1,status:"success",cache:"miss",keptIds:["f1"],injectedIds:["f1"],results:[{id:"f1",text:"fixture memory"}]});
    h.bus.emit("hindsight:retrieval",{version:1,phase:"injection",mode:"automatic",sessionId:"memory-session",contextId:"c1",retrievalIds:["r1"],startedAt:now,endedAt:now,status:"success",cache:"miss",injected:true,renderedHash:"fixture-hash",renderedLength:14});
    await h.emit("before_provider_request",{payload:{messages:[{role:"user",content:"question"}]}});await h.emit("message_end",{message:{role:"assistant",content:[{type:"text",text:"done"}]}});await h.emit("agent_settled");await waitForRequests(capture,1);
    const spans=capture.spans();const retriever=spans.find((s:any)=>s.attrs["langfuse.observation.type"]==="retriever");assert.ok(retriever);assert.equal(metadata(retriever,"bank_id"),"fixture-bank");
    const gen=spans.find((s:any)=>s.name==="LLM Call");assert.equal(metadata(gen,"memory_context_id"),"c1");assert.ok(String(metadata(gen,"memory_retrieval_ids")).includes("r1"));await h.emit("session_shutdown",{reason:"quit"});
  }));
});
