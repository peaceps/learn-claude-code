import { ContentBlock, MessageParam } from '@anthropic-ai/sdk/resources/messages/messages.js';
import { ToolUseBlock } from '@anthropic-ai/sdk/resources';
import { extractText } from '../utils/utils.js';
import { LLMModel } from '../llm/init-llmgw.js';
import { bashTool, ToolCallback, ToolDesc } from '../tools/tools.js';
import { readUserInput } from '../utils/input-reader.js';

type LoopState = {  
    messages: MessageParam[];
    turnCount: number;
    transition_reason?: string
}

const SYSTEM = `You are a coding agent at ${process.cwd()}. "
"Use bash to inspect and change the workspace. Act first, then report clearly."`

export class LoopAgent {
    private llmModel: LLMModel;
    private toolMap: Map<string, ToolCallback> = new Map();
    private history: MessageParam[] = [];
    private builtInTools: ToolDesc[] = [bashTool];

    constructor(system: string = SYSTEM, tools: ToolDesc[] = []) {
        tools = tools.concat(this.builtInTools);
        this.registerTools(tools);
        this.llmModel = new LLMModel(system, tools?.map(t => t.tool));
    }

    private registerTools(tools?: ToolDesc[]): void {
        if (!tools) return;
        for (const tool of tools) {
            this.toolMap.set(tool.tool.name, tool.invoke);
        }
    }

    private async executeToolCalls(contents: ContentBlock[]): Promise<string | null> {
        const results: any[] = [];
        for (const block of contents) {
            if (!this.isToolUse(block)) {
                continue
            }
            const command = (block.input as any)["command"];
            console.log(`\x1b[33m$ ${command}\x1b[0m`);
            const tool = this.toolMap.get(block.name)!;
            const output = await tool(command);
            results.push({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": output,
            })
        }   
        return results.length > 0 ? JSON.stringify(results) : null;
    }

    private isToolUse(content: ContentBlock): content is ToolUseBlock {
        return content.type === "tool_use"
    }

    private async runOneTurn(state: LoopState): Promise<boolean> {
        const response = await this.llmModel.invoke(state.messages);
        state.messages.push({"role": "assistant", "content": response.content});

        if (response.stop_reason != "tool_use") {
            state.transition_reason = '';
            return false
        }

        const results = await this.executeToolCalls(response.content)
        if (!results) {
            state.transition_reason = '';
            return false
        }

        state.messages.push({"role": "user", "content": results})
        state.turnCount += 1
        state.transition_reason = "tool_result"
        return true;
    }

    private async agentLoop(state: LoopState): Promise<void> {
        while (true) {
            const goAound = await this.runOneTurn(state);
            if (!goAound) {
                const finalText = extractText(state.messages[state.messages.length - 1]["content"])
                if (finalText) {
                    console.log(finalText)
                }
                break;
            }
        }
    }

    async invoke(): Promise<void> {
        console.log("输入你想问的内容，或者输入'exit'退出");
        const {read, close} = readUserInput();
        while(true) {
            let query: string = "";
            try {
                query = await read(">>> ");
                if (["q", "exit", ""].includes(query.trim().toLowerCase())) {
                    break
                }
            }
            catch (e){
                console.error("Error reading input:", e);
                break;
            }

            this.history.push({"role": "user", "content": query})
            const state: LoopState = {
                messages: this.history,
                turnCount: 0,
            }
            await this.agentLoop(state)
        }
        console.log("再见！");
        await close(); 
    }
}

