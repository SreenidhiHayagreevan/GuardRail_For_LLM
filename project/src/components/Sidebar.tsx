import { useEffect, useState } from 'react';

interface Conversation {
  id: string;
  title: string;
  messages: { role: 'user' | 'assistant'; text: string }[];
}

interface SidebarProps {
  activeConversationId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onRename: (id: string, title: string) => void;
  onDelete: (id: string) => void;
  onToggleTheme: () => void;
  theme: 'dark' | 'light';
}

export function Sidebar({ activeConversationId, onSelect, onNew, onRename, onDelete, onToggleTheme, theme }: SidebarProps) {
  const [convos, setConvos] = useState<Conversation[]>([]);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editText, setEditText] = useState('');

  useEffect(() => {
    const raw = localStorage.getItem('gr_conversations');
    if (raw) {
      try {
        setConvos(JSON.parse(raw));
      } catch (e) {
        setConvos([]);
      }
    }
  }, []);

  useEffect(() => {
    localStorage.setItem('gr_conversations', JSON.stringify(convos));
  }, [convos]);

  function handleSelect(id: string) {
    onSelect(id);
  }

  function handleNew() {
    const id = String(Date.now());
    const newConvo: Conversation = { id, title: 'New Conversation', messages: [] };
    const next = [newConvo, ...convos];
    setConvos(next);
    onNew();
    onSelect(id);
  }

  function startEdit(c: Conversation) {
    setEditingId(c.id);
    setEditText(c.title);
  }

  function saveEdit(id: string) {
    const next = convos.map((c) => (c.id === id ? { ...c, title: editText || 'Untitled' } : c));
    setConvos(next);
    onRename(id, editText || 'Untitled');
    setEditingId(null);
  }

  function remove(id: string) {
    const next = convos.filter((c) => c.id !== id);
    setConvos(next);
    onDelete(id);
  }

  return (
    <aside className="w-64 bg-gray-800 text-gray-100 p-3 flex-shrink-0">
      <div className="flex items-center justify-between mb-3">
        <div className="font-semibold">Conversations</div>
        <button onClick={handleNew} className="text-xs px-2 py-1 bg-emerald-600 rounded">New</button>
      </div>

      <div className="space-y-2 overflow-auto" style={{ maxHeight: '60vh' }}>
        {convos.length === 0 && <div className="text-sm text-gray-400">No conversations yet</div>}
        {convos.map((c) => (
          <div key={c.id} className={`p-2 rounded cursor-pointer ${c.id === activeConversationId ? 'bg-emerald-700' : 'bg-gray-900'}`}>
            <div className="flex items-center justify-between">
              {editingId === c.id ? (
                <input
                  value={editText}
                  onChange={(e) => setEditText(e.target.value)}
                  className="bg-gray-800 px-2 py-1 rounded text-sm w-full"
                />
              ) : (
                <div onClick={() => handleSelect(c.id)} className="truncate text-sm">{c.title}</div>
              )}
              <div className="flex items-center gap-1 ml-2">
                {editingId === c.id ? (
                  <>
                    <button onClick={() => saveEdit(c.id)} className="text-xs px-2 py-1 bg-emerald-500 rounded">Save</button>
                    <button onClick={() => setEditingId(null)} className="text-xs px-2 py-1 bg-gray-700 rounded">Cancel</button>
                  </>
                ) : (
                  <>
                    <button onClick={() => startEdit(c)} className="text-xs px-2 py-1 bg-gray-700 rounded">Rename</button>
                    <button onClick={() => remove(c.id)} className="text-xs px-2 py-1 bg-red-600 rounded">Del</button>
                  </>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>

      </aside>
  );
}
