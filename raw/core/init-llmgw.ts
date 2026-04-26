import Anthropic from '@anthropic-ai/sdk';

import { llmGateway } from './llmgw-config.json';
import { MessageParam } from '@anthropic-ai/sdk/resources/messages/messages.mjs';
import { ToolUnion } from '@anthropic-ai/sdk/resources.js';

const gw = {
    embeddingModel: llmGateway.embeddingModel ?? 'text-embedding-3-small',
    model: llmGateway.llmgwModel ?? 'GPT5',
    baseURL: llmGateway.llmgwApiBase.replace(/\/+$/, ''),
    apiKey: llmGateway.llmgwApiKey,
    headers: {
        'api-key': llmGateway.llmgwApiKey,
        'workspacename': ('llmgwWorkspace' in llmGateway) ? llmGateway.llmgwWorkspace as string : ''
    },
    timeoutMs: (llmGateway.timeout ?? 300) * 1000, // JSON: seconds → client: ms
    temperature: llmGateway.temperature ?? 0.1,
    maxTokens: llmGateway.maxTokens ?? 8000,
    tavilyApiKey: llmGateway.tavilyApiKey
}

export class LLMModel {
    private client: Anthropic;

    constructor(private system: string = '', private tools?: ToolUnion[]) {
        this.client = new Anthropic({
            apiKey: gw.apiKey, // This is the default and can be omitted
            baseURL: gw.baseURL, // This is the default and can be omitted,
            timeout: gw.timeoutMs
        });
    }

    async invoke(messages: MessageParam[]): Promise<any> {
        return this.client.messages.create({
            model: gw.model,
            system: this.system,
            messages,
            tools: this.tools,
            max_tokens: gw.maxTokens,
            temperature: gw.temperature,
        });
    }
}