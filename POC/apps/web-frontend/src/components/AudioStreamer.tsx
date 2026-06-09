import React, { useState, useRef, useEffect } from 'react';
import { ASRWebSocketClient, type ASRStreamConfig } from '../services/websocket_client';
import { ASRConfigPanel } from './ASRConfigPanel';

export const AudioStreamer: React.FC = () => {
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [readyState, setReadyState] = useState<string>('CHƯA KẾT NỐI');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [liveTranscript, setLiveTranscript] = useState<string>('');
  const [finalTranscript, setFinalTranscript] = useState<string>('');
  const [confidence, setConfidence] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  // ASR Configuration State (US2)
  const [method, setMethod] = useState<'greedy_search' | 'modified_beam_search'>('greedy_search');
  const [beamSize, setBeamSize] = useState<number>(4);
  const [causal, setCausal] = useState<boolean>(true); // Streaming recommends causal decoding

  // Web Audio & WebSocket References
  const wsClientRef = useRef<ASRWebSocketClient | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      stopRecordingAndCleanup();
    };
  }, []);

  const startRecording = async () => {
    setError(null);
    setLiveTranscript('');
    setFinalTranscript('');
    setConfidence(null);
    setSessionId(null);
    setReadyState('ĐANG KẾT NỐI...');

    try {
      // 1. Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        }
      });
      mediaStreamRef.current = stream;

      // 2. Initialize Web Audio API targeting 16000Hz (VietASR native rate)
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      const audioContext = new AudioCtx({ sampleRate: 16000 });
      audioContextRef.current = audioContext;

      sourceRef.current = audioContext.createMediaStreamSource(stream);
      
      // ScriptProcessorNode handles chunk extraction (4096 samples = ~256ms chunk)
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      // 3. Setup WebSocket connection client
      const config: ASRStreamConfig = {
        method,
        causal,
        ...(method === 'modified_beam_search' && { beam_size: beamSize }),
        chunk_size: 16, // Default stream chunk frames
        left_context_frames: 128
      };

      const callbacks = {
        onHandshakeOk: (sid: string) => {
          setSessionId(sid);
          setReadyState('ĐANG NGHE');
          setIsRecording(true);
        },
        onTranscriptUpdate: (text: string, _isFinal: boolean, conf: number) => {
          setLiveTranscript(text);
          setConfidence(conf);
        },
        onFinished: (fullTranscript: string, conf: number) => {
          setFinalTranscript(fullTranscript);
          setLiveTranscript('');
          setConfidence(conf);
          setReadyState('HOÀN THÀNH');
          setIsRecording(false);
          stopRecordingAndCleanup();
        },
        onError: (err: any) => {
          console.error('ASR WS Error:', err);
          setError('Lỗi kết nối WebSocket hoặc máy chủ ASR.');
          setReadyState('LỖI KẾT NỐI');
          setIsRecording(false);
          stopRecordingAndCleanup();
        },
        onClose: (code: number, reason: string) => {
          console.log(`ASR WS Closed: ${code} - ${reason}`);
          if (code === 1008) {
            setError('Kết nối bị từ chối do cấu hình không hợp lệ hoặc hết thời gian handshake.');
          }
          setReadyState('ĐÃ ĐÓNG');
          setIsRecording(false);
          stopRecordingAndCleanup();
        }
      };

      const wsClient = new ASRWebSocketClient(callbacks);
      wsClientRef.current = wsClient;

      // Establish WebSocket
      wsClient.connect(config);

      // 4. Bind Audio Process logic to transcode float32 sample chunks to signed Int16 PCM bytes
      processor.onaudioprocess = (e) => {
        if (wsClient.getReadyState() !== WebSocket.OPEN) return;

        const inputBuffer = e.inputBuffer.getChannelData(0);
        const pcmData = new Int16Array(inputBuffer.length);

        for (let i = 0; i < inputBuffer.length; i++) {
          const s = Math.max(-1.0, Math.min(1.0, inputBuffer[i]));
          pcmData[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }

        // Transmit binary Int16 PCM chunk
        try {
          wsClient.sendAudioChunk(pcmData.buffer);
        } catch (err) {
          console.warn('Failed to transmit audio frame:', err);
        }
      };

      // Connect nodes
      sourceRef.current.connect(processor);
      processor.connect(audioContext.destination);

    } catch (err: any) {
      console.error('Microphone initialization failed:', err);
      setError('Không thể truy cập Microphone. Vui lòng cấp quyền và thử lại.');
      setReadyState('LỖI MICROPHONE');
      setIsRecording(false);
      stopRecordingAndCleanup();
    }
  };

  const stopRecording = () => {
    if (!isRecording) return;
    setReadyState('ĐANG XỬ LÝ...');
    
    // Send Stop request to ASR Server to receive final finished frame
    if (wsClientRef.current) {
      try {
        wsClientRef.current.sendStop();
      } catch (err) {
        console.error('Error sending stop signal:', err);
        stopRecordingAndCleanup();
        setIsRecording(false);
      }
    } else {
      setIsRecording(false);
    }
  };

  const stopRecordingAndCleanup = () => {
    // 1. Disconnect and stop audio processors
    if (processorRef.current) {
      try {
        processorRef.current.disconnect();
      } catch (e) {}
      processorRef.current.onaudioprocess = null;
      processorRef.current = null;
    }

    if (sourceRef.current) {
      try {
        sourceRef.current.disconnect();
      } catch (e) {}
      sourceRef.current = null;
    }

    if (audioContextRef.current) {
      try {
        audioContextRef.current.close();
      } catch (e) {}
      audioContextRef.current = null;
    }

    // 2. Stop microphone stream tracks
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }

    // 3. Shutdown WebSocket
    if (wsClientRef.current) {
      wsClientRef.current.close();
      wsClientRef.current = null;
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto px-4 py-8">
      {/* Configuration Header Card */}
      <div className="backdrop-blur-md bg-slate-900/60 border border-slate-800 rounded-2xl p-6 mb-8 shadow-2xl">
        <h1 className="text-3xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-violet-400 to-indigo-300 mb-2">
          VietASR Real-Time Streaming
        </h1>
        <p className="text-slate-400 text-sm mb-6">
          Bật microphone và truyền âm thanh trực tiếp để nhận dạng giọng nói tiếng Việt thời gian thực.
        </p>

        {/* Decoder Options Subpanel (US2 Panel) */}
        <ASRConfigPanel
          method={method}
          setMethod={setMethod}
          beamSize={beamSize}
          setBeamSize={setBeamSize}
          causal={causal}
          setCausal={setCausal}
        />
      </div>

      {/* Recording Control Card */}
      <div className="backdrop-blur-sm bg-slate-900/40 border border-slate-800 rounded-3xl p-10 text-center relative overflow-hidden shadow-xl">
        {/* Glow behind main button */}
        {isRecording && (
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-48 bg-rose-500/20 rounded-full blur-3xl animate-pulse -z-10"></div>
        )}

        {/* Connection Status Badge */}
        <div className="flex justify-center mb-6">
          <span className={`px-4 py-1.5 rounded-full text-xs font-semibold uppercase tracking-wider flex items-center gap-2 border select-none ${
            readyState === 'ĐANG NGHE' 
              ? 'bg-rose-950/40 border-rose-900/60 text-rose-400' 
              : readyState.includes('LỖI')
                ? 'bg-rose-950/40 border-rose-900/60 text-rose-400'
                : readyState.includes('ĐANG KẾT NỐI')
                  ? 'bg-amber-950/40 border-amber-900/60 text-amber-400'
                  : 'bg-slate-950 border-slate-800 text-slate-400'
          }`}>
            <span className={`w-2.5 h-2.5 rounded-full ${
              readyState === 'ĐANG NGHE' 
                ? 'bg-rose-500 animate-ping' 
                : readyState.includes('ĐANG KẾT NỐI')
                  ? 'bg-amber-500 animate-pulse'
                  : 'bg-slate-600'
            }`}></span>
            Trạng thái: {readyState}
          </span>
        </div>

        {/* Large Button Controller */}
        <div className="flex flex-col items-center justify-center mb-8">
          {!isRecording ? (
            <button
              onClick={startRecording}
              disabled={readyState === 'ĐANG XỬ LÝ...'}
              className="w-24 h-24 bg-gradient-to-tr from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white rounded-full flex items-center justify-center shadow-lg hover:shadow-violet-600/30 active:scale-95 transition-all duration-300"
            >
              <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"></path>
              </svg>
            </button>
          ) : (
            <button
              onClick={stopRecording}
              className="w-24 h-24 bg-rose-600 hover:bg-rose-500 text-white rounded-full flex items-center justify-center shadow-lg hover:shadow-rose-600/30 active:scale-95 transition-all duration-300 animate-pulse"
            >
              <svg className="w-10 h-10" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                <rect x="5" y="5" width="10" height="10" rx="1"></rect>
              </svg>
            </button>
          )}
          
          <p className="text-slate-300 font-medium text-lg mt-4">
            {!isRecording ? 'Click để Bắt đầu Phát trực tuyến' : 'Click để Dừng và Xem kết quả'}
          </p>
          {sessionId && (
            <p className="text-slate-500 text-xs mt-1">Session ID: {sessionId}</p>
          )}
        </div>

        {/* Animated Soundwave Visualizer when Recording */}
        {isRecording && (
          <div className="flex items-center justify-center gap-1.5 h-8 mb-4">
            <span className="w-1 h-3 bg-violet-400 rounded animate-[bounce_0.8s_infinite_-0.2s]"></span>
            <span className="w-1 h-6 bg-violet-500 rounded animate-[bounce_0.8s_infinite_-0.4s]"></span>
            <span className="w-1 h-8 bg-indigo-500 rounded animate-[bounce_0.8s_infinite_-0.6s]"></span>
            <span className="w-1 h-5 bg-violet-500 rounded animate-[bounce_0.8s_infinite_-0.3s]"></span>
            <span className="w-1 h-2 bg-violet-400 rounded animate-[bounce_0.8s_infinite_-0.1s]"></span>
          </div>
        )}
      </div>

      {/* Error Alert Display */}
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

      {/* Incremental Live Transcription Output */}
      {liveTranscript && (
        <div className="backdrop-blur-md bg-slate-900/60 border border-slate-800 rounded-2xl p-8 mt-8 shadow-2xl">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4 mb-4">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              Kết quả thời gian thực (Đang nhận dạng)
            </h2>
            {confidence !== null && (
              <span className="bg-violet-950/40 border border-violet-900/60 px-2 py-1 rounded-lg text-xs text-violet-400">
                Độ tin cậy: {(confidence * 100).toFixed(0)}%
              </span>
            )}
          </div>
          <div className="bg-slate-950/80 border border-slate-900 rounded-xl p-5 text-slate-200 text-lg leading-relaxed font-medium">
            <p className="animate-pulse">{liveTranscript} <span className="inline-block w-1.5 h-4 bg-violet-400 animate-ping"></span></p>
          </div>
        </div>
      )}

      {/* Completed Transcription Results Card */}
      {finalTranscript && (
        <div className="backdrop-blur-md bg-slate-900/60 border border-slate-800 rounded-2xl p-8 mt-8 shadow-2xl">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4 mb-4">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              Kết quả nhận dạng cuối cùng
            </h2>
            {confidence !== null && (
              <span className="bg-emerald-950/40 border border-emerald-900/60 px-2 py-1 rounded-lg text-xs text-emerald-400">
                Độ tin cậy: {(confidence * 100).toFixed(0)}%
              </span>
            )}
          </div>
          <div className="bg-slate-950/80 border border-slate-900 rounded-xl p-5 text-slate-100 text-lg leading-relaxed font-bold">
            <p>{finalTranscript}</p>
          </div>
        </div>
      )}
    </div>
  );
};
