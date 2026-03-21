import { useState, useEffect, useRef } from 'react';
import { ShieldCheck } from 'lucide-react';
import { Sidebar } from './Sidebar';

const MODEL_OPTIONS = [
  { key: 'qwen0.5', label: 'Qwen 0.5' },
  { key: 'qwen2.5', label: 'Qwen 2.5' },
  { key: 'phi3', label: 'Phi-3' },
  { key: 'llama3', label: 'Llama 3' },
  { key: 'gemma3', label: 'Gemma 3' },
  { key: 'mistral', label: 'Mistral' },
];

interface Message {
  role: 'user' | 'assistant';
  text: string;
  result?: any;
}

interface Conversation {
  id: string;
  title: string;
  messages: Message[];
}

// ── Guardrail Analysis Panel ──────────────────────────────────────────────────
function GuardrailAnalysis({ result, theme }: { result: any; theme: 'dark' | 'light' }) {
  const dk = theme === 'dark';
  const cardTitle  = dk ? 'text-gray-400' : 'text-slate-500';
  const cardText   = dk ? 'text-gray-200' : 'text-slate-700';
  const innerBg    = dk ? 'bg-gray-900 text-gray-300' : 'bg-slate-100 text-slate-700';
  const progressBg = dk ? 'bg-gray-700' : 'bg-slate-200';
  const panelBg    = dk ? 'bg-gray-800 border-gray-700' : 'bg-white border-slate-200';
  const card       = dk ? 'bg-gray-900 border border-gray-700' : 'bg-slate-50 border border-slate-200';

  const verdict = result?.guardrails?.input?.rule_based?.valid;
  const checks  = result?.guardrails?.input?.rule_based?.checks || {};
  const ml      = result?.guardrails?.input?.ml_based || null;
  const out     = result?.guardrails?.output || null;
  const meta    = result?.metadata || null;

  return (
    <div className={`mt-2 rounded-xl border shadow-lg overflow-hidden ${panelBg}`}>

      {/* Panel header */}
      <div className={`px-4 py-3 border-b flex items-center gap-2 ${dk ? 'border-gray-700 bg-gray-900' : 'border-slate-200 bg-slate-50'}`}>
        <ShieldCheck size={16} className="text-emerald-500" />
        <span className="text-sm font-semibold">Guardrail Analysis</span>
        <span className={`ml-auto text-xs font-bold px-2 py-0.5 rounded-full ${verdict ? 'bg-emerald-500 text-white' : 'bg-red-500 text-white'}`}>
          {verdict ? 'SAFE' : 'BLOCKED'}
        </span>
      </div>

      {/* ── Row 1: Raw LLM Response | Guarded Response ── */}
      <div className="p-4 grid grid-cols-2 gap-4">
        <div className={`p-3 rounded-lg ${card}`}>
          <h3 className={`text-xs font-semibold uppercase tracking-widest mb-2 ${cardTitle}`}>Raw LLM Response</h3>
          <div className={`text-xs leading-relaxed whitespace-pre-wrap overflow-y-auto ${cardText}`} style={{ maxHeight: '140px' }}>
            {meta?.raw_llm_response}
          </div>
        </div>

        <div className={`p-3 rounded-lg ${card}`}>
          <h3 className={`text-xs font-semibold uppercase tracking-widest mb-2 ${cardTitle}`}>Guarded Response</h3>
          <div className={`text-xs leading-relaxed whitespace-pre-wrap overflow-y-auto ${cardText}`} style={{ maxHeight: '140px' }}>
            {result?.response}
          </div>
        </div>
      </div>

      {/* ── Row 2: Safety Checks | Output Guardrail ── */}
      <div className="px-4 pb-4 grid grid-cols-2 gap-4">
        <div className={`p-3 rounded-lg ${card}`}>
          <h3 className={`text-xs font-semibold uppercase tracking-widest mb-3 ${cardTitle}`}>Safety Checks</h3>
          <div className="grid grid-cols-2 gap-2">
            {['privacy', 'hate', 'violence_illegal', 'misinformation', 'bias', 'prompt_injection'].map((k) => {
              const passed = checks[k]?.passed === true;
              return (
                <div key={k} className={`flex items-center justify-between px-3 py-3 rounded-lg text-sm font-medium ${innerBg}`}>
                  <span className="capitalize">{k.replace('_', ' ')}</span>
                  <span className="text-base">{passed ? '✅' : '❌'}</span>
                </div>
              );
            })}
          </div>
        </div>

        <div className={`p-3 rounded-lg ${card} space-y-4`}>
          {/* ML Check */}
          <div>
            <h3 className={`text-xs font-semibold uppercase tracking-widest mb-2 ${cardTitle}`}>ML Check</h3>
            <p className={`text-xs mb-1.5 ${cardText}`}>
              Unsafe probability: <span className="font-semibold">{((ml?.unsafe_probability ?? 0) * 100).toFixed(4)}%</span>
            </p>
            <div className={`w-full h-1.5 rounded-full overflow-hidden ${progressBg}`}>
              <div className="h-1.5 bg-red-500 rounded-full" style={{ width: `${Math.min(100, (ml?.unsafe_probability ?? 0) * 100)}%` }} />
            </div>
            <div className={`flex justify-between text-xs mt-1 ${cardTitle}`}>
              <span>0% (Safe)</span><span>50% (Threshold)</span><span>100% (Unsafe)</span>
            </div>
          </div>

          {/* Output Guardrail */}
          <div>
            <h3 className={`text-xs font-semibold uppercase tracking-widest mb-2 ${cardTitle}`}>Output Guardrail</h3>
            <p className={`text-xs mb-1.5 ${cardText}`}>
              Hallucination similarity: <span className="font-semibold">{out?.checks?.hallucination_similarity?.toFixed(4) ?? 'N/A'}</span>
            </p>
            <div className={`w-full h-1.5 rounded-full overflow-hidden ${progressBg}`}>
              <div className="h-1.5 bg-emerald-500 rounded-full" style={{ width: `${Math.min(100, (out?.checks?.hallucination_similarity ?? 0) * 100)}%` }} />
            </div>
            <div className={`flex justify-between text-xs mt-1 ${cardTitle}`}>
              <span>0% (Hallucination)</span><span>60% (Threshold)</span><span>100% (Grounded)</span>
            </div>
          </div>

          {/* RAG Metadata */}
          <div>
            <h3 className={`text-xs font-semibold uppercase tracking-widest mb-2 ${cardTitle}`}>RAG Metadata</h3>
            <p className={`text-xs ${cardText}`}>RAG used: <span className="font-semibold">{String(meta?.rag_used)}</span></p>
            <p className={`text-xs ${cardText}`}>Retrieved docs: <span className="font-semibold">{meta?.retrieved_docs_total}</span></p>
            <p className={`text-xs ${cardText}`}>KB sources: <span className="font-semibold">{meta?.kb_sources?.join(', ')}</span></p>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Single chat message ───────────────────────────────────────────────────────
function ChatBubble({ msg, theme }: { msg: Message; theme: 'dark' | 'light' }) {
  const [showAnalysis, setShowAnalysis] = useState(false);
  const dk = theme === 'dark';

  if (msg.role === 'user') {
    return (
      <div className="flex justify-end mb-3">
        <div className="relative max-w-xl">
          <div className="bg-emerald-600 text-white text-sm px-4 py-3 rounded-2xl rounded-tr-none shadow leading-relaxed">
            {msg.text}
          </div>
          <div className="absolute top-0 right-0 w-0 h-0"
            style={{ borderLeft: '10px solid #059669', borderBottom: '10px solid transparent', transform: 'translateX(100%)' }}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-start mb-3">
      <div className="relative max-w-2xl w-full">
        <div className={`text-sm px-4 py-3 rounded-2xl rounded-tl-none shadow leading-relaxed ${dk ? 'bg-gray-700 text-gray-100' : 'bg-white text-slate-800 border border-slate-200'}`}>
          {msg.text}
        </div>
        <div className="absolute top-0 left-0 w-0 h-0"
          style={{
            borderRight: `10px solid ${dk ? '#374151' : '#ffffff'}`,
            borderBottom: '10px solid transparent',
            transform: 'translateX(-100%)',
          }}
        />
      </div>

      {msg.result && (
        <button
          onClick={() => setShowAnalysis((v) => !v)}
          className="mt-1 ml-1 text-xs text-emerald-500 hover:text-emerald-400 underline underline-offset-2 transition-colors"
        >
          {showAnalysis ? 'Hide Details' : 'View Details'}
        </button>
      )}

      {showAnalysis && msg.result && (
        <div className="w-full mt-1">
          <GuardrailAnalysis result={msg.result} theme={theme} />
        </div>
      )}
    </div>
  );
}

// ── Main container ────────────────────────────────────────────────────────────
export function ChatContainer() {
  const [prompt, setPrompt] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [model, setModel] = useState<string>('qwen2.5');
  const [theme, setTheme] = useState<'dark' | 'light'>(() =>
    (localStorage.getItem('gr_theme') as 'dark' | 'light') || 'dark'
  );
  const bottomRef = useRef<HTMLDivElement>(null);

  const [conversations, setConversations] = useState<Conversation[]>(() => {
    try {
      const raw = localStorage.getItem('gr_conversations');
      return raw ? JSON.parse(raw) : [];
    } catch { return []; }
  });

  const [activeConversationId, setActiveConversationId] = useState<string | null>(() => {
    try {
      const raw = localStorage.getItem('gr_conversations');
      const list = raw ? JSON.parse(raw) : [];
      return list.length ? list[0].id : null;
    } catch { return null; }
  });

  const dk = theme === 'dark';
  const pageBg   = dk ? 'bg-gray-900 text-gray-100' : 'bg-slate-100 text-slate-900';
  const inputBg  = dk ? 'bg-gray-800 text-gray-100 border-gray-700' : 'bg-white text-slate-900 border-slate-300';
  const selectBg = dk ? 'bg-gray-800 text-gray-100' : 'bg-white text-slate-800 border border-slate-300';
  const moonBtn  = dk ? 'bg-gray-700 hover:bg-gray-600' : 'bg-slate-200 hover:bg-slate-300';
  const chatBg   = dk ? 'bg-gray-900' : 'bg-slate-50';

  useEffect(() => {
    localStorage.setItem('gr_conversations', JSON.stringify(conversations));
  }, [conversations]);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dk);
    localStorage.setItem('gr_theme', theme);
  }, [theme]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversations, activeConversationId]);

  const activeConversation = conversations.find((c) => c.id === activeConversationId) ?? null;

  function handleNewConversation() {
    const id = String(Date.now());
    setConversations((p) => [{ id, title: 'New Conversation', messages: [] }, ...p]);
    setActiveConversationId(id);
    setPrompt('');
  }

  function handleSelectConversation(id: string) { setActiveConversationId(id); }

  function handleRenameConversation(id: string, title: string) {
    setConversations((p) => p.map((c) => (c.id === id ? { ...c, title } : c)));
  }

  function handleDeleteConversation(id: string) {
    setConversations((p) => p.filter((c) => c.id !== id));
    if (activeConversationId === id) {
      const remaining = conversations.filter((c) => c.id !== id);
      setActiveConversationId(remaining.length ? remaining[0].id : null);
    }
  }

  const submit = async () => {
    if (!prompt.trim()) return;
    setIsLoading(true);
    setError(null);

    let currentId = activeConversationId;
    if (!currentId) {
      const id = String(Date.now());
      setConversations((p) => [{ id, title: prompt.slice(0, 40), messages: [] }, ...p]);
      setActiveConversationId(id);
      currentId = id;
    }

    const sentPrompt = prompt;
    setPrompt('');

    setConversations((prev) =>
      prev.map((c) =>
        c.id === currentId
          ? {
              ...c,
              title: c.title === 'New Conversation' ? sentPrompt.slice(0, 40) : c.title,
              messages: [...c.messages, { role: 'user', text: sentPrompt }],
            }
          : c
      )
    );

    try {
      const resp = await fetch('http://localhost:8000/api/guardrail', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: sentPrompt, model }),
      });
      if (!resp.ok) throw new Error(`Server returned ${resp.status}`);
      const data = await resp.json();

      setConversations((prev) =>
        prev.map((c) =>
          c.id === currentId
            ? {
                ...c,
                messages: [
                  ...c.messages,
                  { role: 'assistant', text: data.response || '', result: data },
                ],
              }
            : c
        )
      );
    } catch (e: any) {
      setError(e.message || String(e));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={`h-screen flex overflow-hidden ${pageBg}`}>

      <Sidebar
        conversations={conversations}
        activeConversationId={activeConversationId}
        onSelect={handleSelectConversation}
        onNew={handleNewConversation}
        onRename={handleRenameConversation}
        onDelete={handleDeleteConversation}
        theme={theme}
      />

      <div className="flex-1 flex flex-col h-screen overflow-hidden">

        {/* Fixed header */}
        <header className={`flex-shrink-0 flex items-center justify-between px-6 py-3 border-b ${dk ? 'bg-gray-900 border-gray-700' : 'bg-slate-100 border-slate-200'}`}>
          <div className="flex items-center gap-3">
            <div className="p-2 bg-emerald-600 rounded-full">
              <ShieldCheck className="text-white" size={18} />
            </div>
            <h1 className="text-xl font-bold">GuardRail LLM</h1>
          </div>
          <div className="flex items-center gap-3">
            <select value={model} onChange={(e) => setModel(e.target.value)} className={`p-2 rounded-lg text-sm ${selectBg}`}>
              {MODEL_OPTIONS.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}
            </select>
            <button onClick={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))} className={`p-2 rounded-lg ${moonBtn}`}>
              {dk ? (
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/></svg>
              ) : (
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#334155" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>
              )}
            </button>
          </div>
        </header>

        {/* Chat area */}
        <div className={`flex-1 overflow-y-auto px-6 py-4 ${chatBg}`}>
          <div className="max-w-4xl mx-auto">

            {(!activeConversation || activeConversation.messages.length === 0) && (
              <div className="flex flex-col items-center justify-center h-64 text-center">
                <div className="p-4 bg-emerald-600 rounded-full mb-4">
                  <ShieldCheck size={28} className="text-white" />
                </div>
                <h2 className={`text-lg font-semibold mb-1 ${dk ? 'text-white' : 'text-slate-800'}`}>GuardRail LLM</h2>
                <p className={`text-sm ${dk ? 'text-gray-400' : 'text-slate-500'}`}>Ask anything. Your prompts are protected by multi-layer guardrails.</p>
              </div>
            )}

            {activeConversation?.messages.map((msg, i) => (
              <ChatBubble key={i} msg={msg} theme={theme} />
            ))}

            {isLoading && (
              <div className="flex items-center gap-2 mb-3">
                <div className={`px-4 py-3 rounded-2xl rounded-tl-none text-sm flex items-center gap-2 ${dk ? 'bg-gray-700 text-gray-300' : 'bg-white text-slate-600 border border-slate-200'}`}>
                  <div className="animate-spin w-3 h-3 border-2 border-t-transparent rounded-full border-emerald-500" />
                  Thinking...
                </div>
              </div>
            )}

            {error && <div className="mb-3 p-3 bg-red-600 text-white rounded-lg text-sm">Error: {error}</div>}

            <div ref={bottomRef} />
          </div>
        </div>

        {/* Fixed input bar */}
        <div className={`flex-shrink-0 border-t px-6 py-4 ${dk ? 'bg-gray-900 border-gray-700' : 'bg-slate-100 border-slate-200'}`}>
          <div className="max-w-4xl mx-auto flex gap-3 items-end">
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); } }}
              placeholder="Type your prompt... (Enter to send, Shift+Enter for new line)"
              rows={2}
              className={`flex-1 p-3 rounded-xl border focus:outline-none focus:ring-2 focus:ring-emerald-500 text-sm resize-none ${inputBg}`}
            />
            <button
              onClick={submit}
              disabled={isLoading || !prompt.trim()}
              className="flex-shrink-0 bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 text-white px-5 py-3 rounded-xl text-sm font-medium"
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
