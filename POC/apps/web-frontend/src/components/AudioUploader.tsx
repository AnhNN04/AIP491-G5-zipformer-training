import React, { useState, useRef } from 'react';
import { uploadAudio, type TranscriptionResponse, type DecodingConfigInput } from '../services/api_client';

export const AudioUploader: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<TranscriptionResponse | null>(null);
  
  // ASR Configuration State (integrating User Story 2 context)
  const [method, setMethod] = useState<'greedy_search' | 'modified_beam_search'>('greedy_search');
  const [beamSize, setBeamSize] = useState<number>(4);
  const [causal, setCausal] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const dragOverRef = useRef<HTMLDivElement>(null);
  const [isDragOver, setIsDragOver] = useState<boolean>(false);

  const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
  const SUPPORTED_EXTENSIONS = ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.mp4'];

  const validateAndSetFile = (selectedFile: File) => {
    setError(null);
    setResult(null);

    const extension = '.' + selectedFile.name.split('.').pop()?.toLowerCase();
    if (!SUPPORTED_EXTENSIONS.includes(extension)) {
      setError(`Format file không hỗ trợ. Chỉ hỗ trợ các định dạng: ${SUPPORTED_EXTENSIONS.join(', ')}`);
      setFile(null);
      return;
    }

    if (selectedFile.size > MAX_FILE_SIZE) {
      setError('Dung lượng file vượt quá giới hạn cho phép (Tối đa 50MB).');
      setFile(null);
      return;
    }

    setFile(selectedFile);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleBrowseClick = () => {
    fileInputRef.current?.click();
  };

  const handleUpload = async () => {
    if (!file) return;
    
    setIsUploading(true);
    setError(null);
    setResult(null);

    const config: DecodingConfigInput = {
      method,
      causal,
      ...(method === 'modified_beam_search' && { beam_size: beamSize }),
    };

    try {
      const response = await uploadAudio(file, config);
      setResult(response);
    } catch (err) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('Đã xảy ra lỗi kết nối mạng không xác định.');
      }
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto px-4 py-8">
      {/* Configuration Header Card */}
      <div className="backdrop-blur-md bg-slate-900/60 border border-slate-800 rounded-2xl p-6 mb-8 shadow-2xl">
        <h1 className="text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-violet-400 to-indigo-300 mb-2">
          VietASR Batch Speech Recognition
        </h1>
        <p className="text-slate-400 text-sm mb-6">
          Tải lên các tệp tin âm thanh hoặc video để chuyển đổi sang văn bản tiếng Việt.
        </p>

        {/* Decoder Options Subpanel (US2 Panel) */}
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
      </div>

      {/* Drag & Drop Upload Zone */}
      <div
        ref={dragOverRef}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={!file && !isUploading ? handleBrowseClick : undefined}
        className={`relative backdrop-blur-sm border-2 border-dashed rounded-3xl p-10 text-center transition-all duration-300 cursor-pointer shadow-xl ${
          isDragOver 
            ? 'border-violet-500 bg-violet-500/10 scale-102' 
            : file 
              ? 'border-slate-700 bg-slate-900/20 cursor-default' 
              : 'border-slate-800 bg-slate-900/40 hover:border-slate-700 hover:bg-slate-900/30'
        }`}
      >
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept={SUPPORTED_EXTENSIONS.join(',')}
          className="hidden"
        />

        {!file ? (
          <div className="flex flex-col items-center">
            <div className="w-16 h-16 bg-violet-900/20 rounded-full flex items-center justify-center mb-4 text-violet-400 group-hover:scale-110 transition-transform">
              <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
              </svg>
            </div>
            <p className="text-slate-300 font-medium mb-1 text-lg">
              Kéo thả tệp tin của bạn vào đây hoặc <span className="text-violet-400 font-semibold underline">Chọn tệp</span>
            </p>
            <p className="text-slate-500 text-xs mt-2">
              Hỗ trợ: WAV, MP3, FLAC, M4A, OGG, MP4 (Tối đa 50MB)
            </p>
          </div>
        ) : (
          <div className="flex flex-col items-center">
            <div className="w-16 h-16 bg-indigo-900/20 rounded-full flex items-center justify-center mb-4 text-indigo-400">
              <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3"></path>
              </svg>
            </div>
            <p className="text-indigo-200 font-semibold text-lg truncate max-w-lg mb-1">
              {file.name}
            </p>
            <p className="text-slate-500 text-xs">
              Size: {(file.size / (1024 * 1024)).toFixed(2)} MB
            </p>
            
            {!isUploading && (
              <div className="flex gap-4 mt-6">
                <button
                  onClick={handleBrowseClick}
                  className="px-5 py-2 border border-slate-700 hover:border-slate-600 text-slate-400 hover:text-slate-300 rounded-xl text-sm font-medium transition"
                >
                  Chọn tệp khác
                </button>
                <button
                  onClick={handleUpload}
                  className="px-6 py-2 bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white rounded-xl text-sm font-semibold shadow-lg hover:shadow-violet-600/20 transition-all duration-300"
                >
                  Nhận dạng giọng nói
                </button>
              </div>
            )}
          </div>
        )}

        {/* Loading Spinner Overlays */}
        {isUploading && (
          <div className="absolute inset-0 bg-slate-950/80 rounded-3xl flex flex-col items-center justify-center transition-opacity">
            <div className="animate-spin rounded-full h-12 w-12 border-t-4 border-b-4 border-violet-500 mb-4"></div>
            <p className="text-slate-300 font-semibold text-lg">Đang tải lên và xử lý âm thanh...</p>
            <p className="text-slate-500 text-xs mt-1">Có thể mất vài giây tùy thuộc vào độ dài tệp</p>
          </div>
        )}
      </div>

      {/* Error Card Notification */}
      {error && (
        <div className="bg-rose-950/40 border border-rose-900/60 rounded-2xl p-4 mt-6 text-rose-200 text-sm flex items-start shadow-md">
          <svg className="w-5 h-5 text-rose-400 mr-3 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path>
          </svg>
          <div>
            <span className="font-semibold">Lỗi xảy ra: </span>
            {error}
          </div>
        </div>
      )}

      {/* Structured Result Showcase Container */}
      {result && (
        <div className="backdrop-blur-md bg-slate-900/60 border border-slate-800 rounded-2xl p-8 mt-8 shadow-2xl animate-fade-in">
          {/* Metadata Grid */}
          <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-6 mb-6">
            <div className="flex gap-6">
              <div>
                <span className="block text-xs uppercase tracking-wider text-slate-500 font-semibold mb-1">Thời lượng</span>
                <span className="text-slate-200 font-bold">{result.duration_seconds.toFixed(2)}s</span>
              </div>
              <div>
                <span className="block text-xs uppercase tracking-wider text-slate-500 font-semibold mb-1">Dung lượng tệp</span>
                <span className="text-slate-200 font-bold">{(result.size_bytes / 1024).toFixed(1)} KB</span>
              </div>
            </div>

            <div className="flex gap-4">
              <div className="bg-emerald-950/40 border border-emerald-900/60 px-3 py-1.5 rounded-xl text-xs text-emerald-400 flex items-center font-medium">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-2 animate-pulse"></span>
                Hệ số tự tin: {(result.transcription.confidence * 100).toFixed(1)}%
              </div>
              
              <div className="bg-indigo-950/40 border border-indigo-900/60 px-3 py-1.5 rounded-xl text-xs text-indigo-400 flex items-center font-medium">
                Ngữ điệu: <span className="font-bold text-indigo-300 ml-1">{result.dialect.inferred}</span> 
                <span className="text-indigo-500 ml-1">({(result.dialect.probability * 100).toFixed(0)}%)</span>
              </div>
            </div>
          </div>

          {/* Transcription Header */}
          <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3">
            Văn bản nhận dạng (Rê chuột lên từng từ để xem căn biên thời gian)
          </h2>

          {/* Interactive Word Level Alignment Output */}
          <div className="bg-slate-950/80 border border-slate-900 rounded-xl p-5 text-slate-200 text-lg leading-relaxed font-medium">
            {result.transcription.word_alignments && result.transcription.word_alignments.length > 0 ? (
              <div className="flex flex-wrap gap-x-1 gap-y-2">
                {result.transcription.word_alignments.map((w, index) => (
                  <span
                    key={index}
                    className="relative group cursor-help px-1 py-0.5 rounded hover:bg-violet-500/20 hover:text-violet-300 transition-colors duration-150"
                  >
                    {w.word}
                    {/* Tooltip Popup */}
                    <span className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 hidden group-hover:block bg-slate-950 border border-slate-800 text-xs text-slate-300 px-3 py-2 rounded-xl shadow-2xl z-20 pointer-events-none whitespace-nowrap">
                      <div className="font-bold text-violet-400 mb-1">{w.word}</div>
                      <div>Căn biên: {w.start.toFixed(2)}s - {w.end.toFixed(2)}s</div>
                      <div>Độ tin cậy: {(w.conf * 100).toFixed(0)}%</div>
                    </span>
                  </span>
                ))}
              </div>
            ) : (
              <p>{result.transcription.text}</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
