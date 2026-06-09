import { AudioUploader } from './components/AudioUploader';

function App() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-violet-500/30 selection:text-violet-200">
      {/* Glow Blur Background Elements */}
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-violet-600/10 rounded-full blur-3xl pointer-events-none"></div>
      <div className="absolute bottom-10 right-1/4 w-96 h-96 bg-indigo-600/10 rounded-full blur-3xl pointer-events-none"></div>

      {/* Main Navbar */}
      <header className="border-b border-slate-900 backdrop-blur-md bg-slate-950/70 sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 bg-gradient-to-tr from-violet-600 to-indigo-600 rounded-xl flex items-center justify-center font-bold text-white shadow-md shadow-violet-600/20">
              V
            </div>
            <span className="font-bold text-lg tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-slate-100 to-slate-400">
              VietASR POC
            </span>
          </div>

          <nav className="flex items-center gap-4 text-xs font-semibold uppercase tracking-wider text-slate-400">
            <span className="bg-slate-900 border border-slate-800 text-slate-300 px-3 py-1 rounded-full flex items-center gap-2 select-none">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              Gateway Online
            </span>
          </nav>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 py-10 relative">
        <AudioUploader />
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 py-6 text-center text-xs text-slate-600 bg-slate-950">
        <p>© 2026 VietASR Proof of Concept project. All rights reserved.</p>
      </footer>
    </div>
  );
}

export default App;
