export interface ASRStreamConfig {
  method: 'greedy_search' | 'modified_beam_search';
  beam_size?: number;
  causal?: boolean;
  chunk_size?: number;
  left_context_frames?: number;
}

export interface ASRStreamCallbacks {
  onHandshakeOk?: (sessionId: string) => void;
  onTranscriptUpdate?: (text: string, isFinal: boolean, confidence: number) => void;
  onFinished?: (fullTranscript: string, confidence: number) => void;
  onError?: (error: Event | Error) => void;
  onClose?: (code: number, reason: string) => void;
}

export class ASRWebSocketClient {
  private ws: WebSocket | null = null;
  private url: string;
  private callbacks: ASRStreamCallbacks;

  constructor(callbacks: ASRStreamCallbacks, customUrl?: string) {
    this.callbacks = callbacks;
    if (customUrl) {
      this.url = customUrl;
    } else {
      // Resolve websocket URL based on VITE_API_URL or window.location
      const apiBaseUrl = (import.meta.env.VITE_API_URL as string) || 'http://localhost:8000';
      const wsBaseUrl = apiBaseUrl.replace(/^http/, 'ws');
      this.url = `${wsBaseUrl}/api/stream`;
    }
  }

  /**
   * Establishes a WebSocket connection and transmits the handshake config frame.
   */
  public connect(config: ASRStreamConfig): void {
    if (this.ws) {
      this.close();
    }

    try {
      this.ws = new WebSocket(this.url);
    } catch (err) {
      if (this.callbacks.onError) {
        this.callbacks.onError(err as Error);
      }
      return;
    }

    this.ws.binaryType = 'arraybuffer';

    this.ws.onopen = () => {
      const handshakeFrame = {
        event: 'handshake',
        config
      };
      this.ws?.send(JSON.stringify(handshakeFrame));
    };

    this.ws.onmessage = (event) => {
      if (typeof event.data !== 'string') {
        return;
      }

      try {
        const message = JSON.parse(event.data);
        switch (message.event) {
          case 'handshake_ok':
            if (this.callbacks.onHandshakeOk) {
              this.callbacks.onHandshakeOk(message.session_id);
            }
            break;
          case 'transcript_update':
            if (this.callbacks.onTranscriptUpdate) {
              this.callbacks.onTranscriptUpdate(
                message.text,
                message.is_final || false,
                message.confidence ?? 0.0
              );
            }
            break;
          case 'finished':
            if (this.callbacks.onFinished) {
              this.callbacks.onFinished(
                message.full_transcript,
                message.confidence ?? 0.0
              );
            }
            break;
          default:
            console.warn('Unknown ASR event:', message.event);
        }
      } catch (err) {
        if (this.callbacks.onError) {
          this.callbacks.onError(err as Error);
        }
      }
    };

    this.ws.onerror = (err) => {
      if (this.callbacks.onError) {
        this.callbacks.onError(err);
      }
    };

    this.ws.onclose = (event) => {
      if (this.callbacks.onClose) {
        this.callbacks.onClose(event.code, event.reason);
      }
      this.ws = null;
    };
  }

  /**
   * Transmits a binary PCM audio chunk to the ASR server.
   */
  public sendAudioChunk(chunk: ArrayBuffer | Blob): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket is not open');
    }
    this.ws.send(chunk);
  }

  /**
   * Sends the stop signal to request final evaluation and closing of the connection.
   */
  public sendStop(): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket is not open');
    }
    this.ws.send(JSON.stringify({ event: 'stop' }));
  }

  /**
   * Forcefully closes the client WebSocket session.
   */
  public close(): void {
    if (this.ws) {
      // Remove event listeners to prevent callbacks firing during cleanup
      this.ws.onopen = null;
      this.ws.onmessage = null;
      this.ws.onerror = null;
      this.ws.onclose = null;
      
      try {
        this.ws.close();
      } catch (err) {
        // ignore
      }
      this.ws = null;
    }
  }

  /**
   * Retrieves the current ready state of the WebSocket.
   */
  public getReadyState(): number {
    return this.ws ? this.ws.readyState : WebSocket.CLOSED;
  }
}
