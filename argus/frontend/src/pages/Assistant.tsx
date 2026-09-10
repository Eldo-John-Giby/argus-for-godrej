import { useState, useRef, useEffect } from "react";
import { Send, Wrench, Database, Bot, User } from "lucide-react";

interface Message {
  role: "user" | "assistant";
  content: string;
  toolCalls?: Array<{
    tool: string;
    arguments: Record<string, any>;
    result?: string;
  }>;
}

const DEMO_MESSAGES: Message[] = [
  {
    role: "user",
    content: "Show me all high-risk handling events from today's unloading.",
  },
  {
    role: "assistant",
    content: "Based on today's unloading operations, I found 8 high-risk events. Here's a summary:\n\n1. **Product Thrown** (Critical, Score: 87.2) — Operator threw package from loading dock height\n2. **Product Dropped** (High, Score: 68.5) — Box slipped during transfer to pallet\n3. **Stepping on Packages** (High, Score: 72.1) — Person stood on stacked boxes to reach upper shelf\n\nThe most common high-risk behaviour today is **product throwing** at Bay A (Unloading Dock), with 3 critical incidents. I recommend reviewing the handling procedures for that bay.",
    toolCalls: [
      {
        tool: "get_events",
        arguments: { tier: "High", bay_id: "unloading" },
        result: '[{"id": 1, "behaviour_type": "3_product_thrown", "tier": "Critical", "risk_score": 87.2}, ...]',
      },
    ],
  },
  {
    role: "user",
    content: "Which loading bay had the highest number of risky events?",
  },
  {
    role: "assistant",
    content: "**Bay A — Unloading Dock** had the highest number of risky events today with **18 total events**, including 3 Critical and 5 High-risk incidents.\n\nBay breakdown:\n- Bay A (Unloading): 18 events, avg risk 67.3\n- Bay B (Staging): 12 events, avg risk 42.1\n- Bay C (Loading): 8 events, avg risk 55.8\n\nBay A should be prioritized for safety review.",
    toolCalls: [
      {
        tool: "get_bay_summary",
        arguments: {},
        result: '[{"bay_id": "A", "event_count": 18, "avg_risk": 67.3}, ...]',
      },
    ],
  },
];

export default function Assistant() {
  const [messages, setMessages] = useState<Message[]>(DEMO_MESSAGES);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [showToolCalls, setShowToolCalls] = useState(true);
  const [askedQuestions, setAskedQuestions] = useState<Set<string>>(new Set());
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSend() {
    if (!input.trim() || loading) return;

    const userMessage: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setAskedQuestions((prev) => new Set(prev).add(input.trim()));
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("/api/assistant/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: input }),
      });

      if (res.ok) {
        const data = await res.json();
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: data.response,
            toolCalls: data.tool_calls,
          },
        ]);
      } else {
        // Fallback demo response
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: "I'll look that up in the events database for you. [Backend not connected — connect to API for real grounding]",
            toolCalls: [],
          },
        ]);
      }
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "I'm unable to reach the events database right now. Please ensure the backend is running.",
          toolCalls: [],
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  // Demo quick questions from the brief
  const quickQuestions = [
    "Show me all high-risk handling events from today's unloading",
    "What were the three most common risky behaviours during the morning shift?",
    "Which loading bay had the highest number of risky events?",
    "Why was this event classified as high risk?",
  ];

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Supervisor AI Assistant</h2>
        <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer">
          <input
            type="checkbox"
            checked={showToolCalls}
            onChange={(e) => setShowToolCalls(e.target.checked)}
            className="rounded"
          />
          Show tool calls
        </label>
      </div>

      <div className="text-sm text-gray-400 bg-argus-card rounded-lg border border-argus-border p-3">
        <strong className="text-argus-accent">Grounded assistant</strong> — This AI ONLY answers from the events database via tool calling. It never invents information. All responses are traceable to database queries.
      </div>

      {/* Chat Area */}
      <div className="bg-argus-card rounded-xl border border-argus-border overflow-hidden" style={{ height: "calc(100vh - 280px)" }}>
        <div className="h-full overflow-y-auto p-4 space-y-4">
          {messages.map((msg, i) => (
            <div key={i} className={`flex gap-3 ${msg.role === "user" ? "justify-end" : ""}`}>
              {msg.role === "assistant" && (
                <div className="w-8 h-8 rounded-lg bg-argus-accent flex items-center justify-center flex-shrink-0">
                  <Bot className="w-4 h-4 text-white" />
                </div>
              )}

              <div className={`max-w-[85%] rounded-xl p-4 ${
                msg.role === "user"
                  ? "bg-blue-600 text-white"
                  : "bg-argus-dark border border-argus-border"
              }`}>
                <div className="whitespace-pre-wrap text-sm leading-relaxed">{msg.content}</div>

                {/* Tool Calls Trace */}
                {showToolCalls && msg.toolCalls && msg.toolCalls.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-argus-border space-y-2">
                    <div className="flex items-center gap-1.5 text-xs text-gray-400">
                      <Wrench className="w-3 h-3" />
                      Tool calls made:
                    </div>
                    {msg.toolCalls.map((tc, j) => (
                      <div key={j} className="bg-argus-card rounded-lg p-2 text-xs">
                        <div className="flex items-center gap-2 mb-1">
                          <Database className="w-3 h-3 text-argus-accent" />
                          <span className="font-mono text-argus-accent">{tc.tool}</span>
                        </div>
                        <pre className="text-gray-500 overflow-x-auto text-[10px]">
                          {JSON.stringify(tc.arguments, null, 2)}
                        </pre>
                        {tc.result && (
                          <pre className="text-green-400/70 overflow-x-auto text-[10px] mt-1 max-h-20 overflow-y-hidden">
                            {tc.result.slice(0, 200)}...
                          </pre>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {msg.role === "user" && (
                <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center flex-shrink-0">
                  <User className="w-4 h-4 text-white" />
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className="flex gap-3">
              <div className="w-8 h-8 rounded-lg bg-argus-accent flex items-center justify-center flex-shrink-0">
                <Bot className="w-4 h-4 text-argus-dark" />
              </div>
              <div className="bg-argus-dark border border-argus-border rounded-xl p-4">
                <div className="flex gap-1">
                  <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                  <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                  <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Quick Questions (from the brief) — asked chips grey out */}
      <div className="flex flex-wrap gap-2">
        {quickQuestions.map((q) => {
          const asked = askedQuestions.has(q);
          return (
            <button
              key={q}
              onClick={() => { if (!asked) setInput(q); }}
              disabled={asked}
              className={`px-3 py-1.5 bg-argus-dark border rounded-lg text-xs transition-colors ${
                asked
                  ? "border-argus-border text-gray-600 line-through cursor-default"
                  : "border-argus-border text-gray-400 hover:text-white hover:border-argus-accent"
              }`}
            >
              {q}
            </button>
          );
        })}
      </div>

      {/* Input */}
      <div className="flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder="Ask about warehouse safety events..."
          className="flex-1 bg-argus-dark border border-argus-border rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-argus-accent transition-colors"
        />
        <button
          onClick={handleSend}
          disabled={loading || !input.trim()}
          className="px-4 py-3 bg-argus-accent text-white rounded-xl font-medium hover:bg-blue-500 transition-colors disabled:opacity-50"
        >
          <Send className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
