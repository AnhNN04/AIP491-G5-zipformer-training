import React from 'react';

export interface ASRConfigPanelProps {
  method: 'greedy_search' | 'modified_beam_search';
  setMethod: (method: 'greedy_search' | 'modified_beam_search') => void;
  beamSize: number;
  setBeamSize: (size: number) => void;
  causal: boolean;
  setCausal: (causal: boolean) => void;
}

export const ASRConfigPanel: React.FC<ASRConfigPanelProps> = ({
  method,
  setMethod,
  beamSize,
  setBeamSize,
  causal,
  setCausal
}) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-4 border-t border-slate-800">
      <div>
        <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">
          Decoding Search Algorithm
        </label>
        <select
          value={method}
          onChange={(e) => setMethod(e.target.value as any)}
          className="w-full bg-slate-950 border border-slate-800 text-slate-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-violet-500 transition-colors"
        >
          <option value="greedy_search">Greedy Search (Nhanh)</option>
          <option value="modified_beam_search">Modified Beam Search (Chính xác)</option>
        </select>
      </div>

      <div>
        <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">
          Beam Search Size
        </label>
        <input
          type="range"
          min="1"
          max="20"
          value={beamSize}
          disabled={method === 'greedy_search'}
          onChange={(e) => setBeamSize(parseInt(e.target.value))}
          className="w-full accent-violet-500 h-2 bg-slate-950 rounded-lg appearance-none cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed"
        />
        <div className="flex justify-between text-xs text-slate-600 mt-1">
          <span>Size: {beamSize}</span>
          <span>(Tối đa: 20)</span>
        </div>
      </div>

      <div className="flex items-center pt-6">
        <label className="flex items-center cursor-pointer text-sm text-slate-300 select-none">
          <input
            type="checkbox"
            checked={causal}
            onChange={(e) => setCausal(e.target.checked)}
            className="w-4 h-4 rounded border-slate-800 text-violet-600 focus:ring-violet-500 bg-slate-950 mr-3"
          />
          Causal Decoding (Luồng Nhân Quả)
        </label>
      </div>
    </div>
  );
};
