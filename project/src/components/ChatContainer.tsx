import { useState, useEffect } from 'react';
import { ShieldCheck, ShieldOff } from 'lucide-react';
import { ChatInput } from './ChatInput';
import { Sidebar } from './Sidebar';

const MODEL_OPTIONS = [
  { key: 'qwen0.5', label: 'Qwen 0.5' },
  { key: 'qwen2.5', label: 'Qwen 2.5' },
  { key: 'phi3', label: 'Phi-3' },
  { key: 'llama3', label: 'Llama 3' },
  { key: 'gemma3', label: 'Gemma 3' },
  { key: 'mistral', label: 'Mistral' },
];

export function ChatContainer() {
  const [prompt, setPrompt] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [model, setModel] = useState<string>('qwen2.5');

  // Conversations & theme
  const [conversations, setConversations] = useState<any[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [theme, setTheme] = useState<'dark' | 'light'>(() => (localStorage.getItem('gr_theme') as 'dark' | 'light') || 'dark');

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
    localStorage.setItem('gr_theme', theme);
  }, [theme]);

  useEffect(() => {
    const raw = localStorage.getItem('gr_conversations');
    if (raw) {
      try {
        const list = JSON.parse(raw);
        setConversations(list);
        if (list.length) setActiveConversationId(list[0].id);
      } catch (e) {
        setConversations([]);
      }
    }
  }, []);

  useEffect(() => {
    localStorage.setItem('gr_conversations', JSON.stringify(conversations));
  }, [conversations]);

  function appendMessageToActive(role: 'user' | 'assistant', text: string) {
    if (!activeConversationId) return;
    setConversations((prev) => prev.map((c) => (c.id === activeConversationId ? { ...c, messages: [...c.messages, { role, text }] } : c)));
  }

  function handleSelectConversation(id: string) {
    setActiveConversationId(id);
    const convo = conversations.find((c) => c.id === id);
    if (convo) {
      // Optionally load last prompt into input
    }
  }

  function handleNewConversation() {
    const id = String(Date.now());
    const convo = { id, title: 'New Conversation', messages: [] };
    setConversations((p) => [convo, ...p]);
    setActiveConversationId(id);
  }

  function handleRenameConversation(id: string, title: string) {
    setConversations((p) => p.map((c) => (c.id === id ? { ...c, title } : c)));
  }

  function handleDeleteConversation(id: string) {
    setConversations((p) => p.filter((c) => c.id !== id));
    if (activeConversationId === id) setActiveConversationId(null);
  }

  const submit = async (p?: string) => {
    const toSend = p ?? prompt;
    if (!toSend) return;
    setIsLoading(true);
    setError(null);
    setResult(null);

    // Append user message
    if (!activeConversationId) handleNewConversation();
    appendMessageToActive('user', toSend);

    try {
      const resp = await fetch('http://localhost:8000/api/guardrail', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: toSend, model }),
      });

      if (!resp.ok) throw new Error(`Server returned ${resp.status}`);
      const data = await resp.json();
      setResult(data);
      appendMessageToActive('assistant', data.response || '');
    } catch (e: any) {
      setError(e.message || String(e));
    } finally {
      setIsLoading(false);
    }
  };

  const verdict = result?.guardrails?.input?.rule_based?.valid;
  const checks = result?.guardrails?.input?.rule_based?.checks || {};
  const ml = result?.guardrails?.input?.ml_based || null;
  const out = result?.guardrails?.output || null;
  const meta = result?.metadata || null;

  return (
    <div className="min-h-screen flex bg-gray-900 text-gray-100">
      <Sidebar
        activeConversationId={activeConversationId}
        onSelect={handleSelectConversation}
        onNew={handleNewConversation}
        onRename={handleRenameConversation}
        onDelete={handleDeleteConversation}
        onToggleTheme={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
        theme={theme}
      />

      <div className="flex-1 max-w-6xl mx-auto p-6">
        <header className="flex items-center gap-4 mb-6 justify-between">
          <div className="flex items-center gap-4">
            <div className="p-2 bg-emerald-600 rounded-full">
              <ShieldCheck className="text-white" />
            </div>
            <h1 className="text-2xl font-bold">GuardRail LLM</h1>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
              aria-label="Toggle theme"
              className="p-2 rounded bg-gray-700 hover:bg-gray-600"
            >
              {/* Simple moon/sun svg that flips based on theme */}
              {theme === 'dark' ? (
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"></path></svg>
              ) : (
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="black" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="5"></circle><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>
              )}
            </button>
          </div>
        </header>

        <section className="mb-4">
          <div className="flex items-center gap-3 mb-3">
            <label className="text-sm">Model:</label>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="bg-gray-800 text-gray-100 p-2 rounded"
            >
              {MODEL_OPTIONS.map((m) => (
                <option key={m.key} value={m.key}>{m.label}</option>
              ))}
            </select>
          </div>

          <ChatInput
            value={prompt}
            onChange={setPrompt}
            onSubmit={() => submit()}
            disabled={isLoading}
          />
        </section>

        <section className="mb-4">
          {isLoading ? (
            <div className="flex items-center gap-2 text-sm text-gray-300">
              <div className="animate-spin w-4 h-4 border-2 border-t-transparent rounded-full border-gray-400" />
              Sending...
            </div>
          ) : null}

          {result && (
            <div className={`mt-4 p-4 rounded ${verdict ? 'bg-emerald-800' : 'bg-red-800'}`}>
              <div className="flex items-center gap-3">
                <div className="p-2 rounded bg-black/20">
                  {verdict ? <span className="text-emerald-300">✅ SAFE</span> : <span className="text-red-300">🚫 BLOCKED</span>}
                </div>
                <div>
                  <div className="text-sm text-gray-200">Verdict</div>
                  <div className="font-semibold">{verdict ? 'Allowed' : 'Blocked'}</div>
                </div>
              </div>
            </div>
          )}

          {error && <div className="mt-4 p-3 bg-red-900 rounded">Error: {error}</div>}
        </section>

        {result && (
          <main className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-4">
              <div className="p-4 bg-gray-800 rounded">
                <h2 className="font-semibold mb-2">Raw LLM Response</h2>
                <div className="text-sm text-gray-200 whitespace-pre-wrap">{meta?.raw_llm_response}</div>
              </div>
              <div className="p-4 bg-gray-800 rounded">
                <h2 className="font-semibold mb-2">Safety Checks</h2>
                <div className="grid grid-cols-2 gap-2">
                  {['privacy', 'hate', 'violence_illegal', 'misinformation', 'bias', 'prompt_injection'].map((k) => {
                    const passed = checks[k]?.passed === true;
                    return (
                      <div key={k} className="flex items-center justify-between bg-gray-900 p-2 rounded">
                        <div className="capitalize text-sm">{k.replace('_', ' ')}</div>
                        <div className="text-sm">{passed ? '✅' : '❌'}</div>
                      </div>
                    );
                  })}
                </div>

                <div className="mt-4">
                  <h3 className="font-medium mb-2">ML Check</h3>
                  <div className="text-xs text-gray-300 mb-1">Unsafe probability: {(ml?.unsafe_probability ?? 0) * 100}%</div>
                  <div className="w-full bg-gray-700 h-3 rounded overflow-hidden">
                    <div className="h-3 bg-red-500" style={{ width: `${Math.min(100, (ml?.unsafe_probability ?? 0) * 100)}%` }} />
                  </div>
                </div>
              </div>

              <div className="p-4 bg-gray-800 rounded">
                <h2 className="font-semibold mb-2">RAG Metadata</h2>
                <div className="text-sm text-gray-200">RAG used: {String(meta?.rag_used)}</div>
                <div className="text-sm text-gray-200">Retrieved docs total: {meta?.retrieved_docs_total}</div>
                <div className="text-sm text-gray-200">KB sources: {meta?.kb_sources?.join(', ')}</div>
              </div>
            </div>

            <div className="space-y-4">
              <div className="p-4 bg-gray-800 rounded">
                <h2 className="font-semibold mb-2">Guarded Response</h2>
                <div className="text-sm text-gray-200 whitespace-pre-wrap">{result?.response}</div>
              </div>

              <div className="p-4 bg-gray-800 rounded">
                <h2 className="font-semibold mb-2">Output Guardrail</h2>
                <div className="text-sm text-gray-200 mb-2">Hallucination similarity: {out?.checks?.hallucination_similarity ?? 'N/A'}</div>
                <div className="w-full bg-gray-700 h-3 rounded overflow-hidden">
                  <div className="h-3 bg-emerald-500" style={{ width: `${Math.min(100, (out?.checks?.hallucination_similarity ?? 0) * 100)}%` }} />
                </div>
              </div>

              <div className="p-4 bg-gray-800 rounded">
                <h2 className="font-semibold mb-2">Metadata</h2>
                <pre className="text-xs text-gray-200 whitespace-pre-wrap">{JSON.stringify(meta, null, 2)}</pre>
              </div>
            </div>
          </main>
        )}
      </div>
    </div>
  );
}
