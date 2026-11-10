/**
 * Longinfer TypeScript 客户端（零依赖，fetch 实现）。
 */

export interface ChatRequest {
  prompt?: string;
  messages?: Array<{ role: string; content: string }>;
  max_new?: number;
  temperature?: number;
}

export interface ChatResponse {
  text: string;
  provider: string;
  model: string;
  usage: Record<string, unknown>;
}

export interface HealthResponse {
  status: string;
  version: string;
  provider: string;
  model: string;
}

export interface AgentResponse {
  goal: string;
  steps: number;
  trajectory: unknown[];
}

export class LonginferClient {
  private baseUrl: string;
  private token?: string;

  constructor(baseUrl = "http://127.0.0.1:8000", token?: string) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.token = token;
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = { "Content-Type": "application/json" };
    if (this.token) h["Authorization"] = `Bearer ${this.token}`;
    return h;
  }

  async health(): Promise<HealthResponse> {
    const r = await fetch(`${this.baseUrl}/health`);
    return r.json();
  }

  async chat(prompt: string, maxNew = 256, temperature = 0.7): Promise<ChatResponse> {
    const r = await fetch(`${this.baseUrl}/v1/chat`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({ prompt, max_new: maxNew, temperature }),
    });
    return r.json();
  }

  async agentRun(goal: string, maxSteps = 200): Promise<AgentResponse> {
    const r = await fetch(`${this.baseUrl}/v1/agent/run`, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify({ goal, max_steps: maxSteps }),
    });
    return r.json();
  }
}
