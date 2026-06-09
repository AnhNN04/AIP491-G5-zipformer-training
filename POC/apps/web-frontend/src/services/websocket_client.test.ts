import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { ASRWebSocketClient, type ASRStreamCallbacks } from './websocket_client'

// Simple mock for browser WebSocket class
class MockWebSocket {
  public static CONNECTING = 0;
  public static OPEN = 1;
  public static CLOSING = 2;
  public static CLOSED = 3;
  
  public static instances: MockWebSocket[] = [];

  public url: string;
  public binaryType: string = 'blob';
  public readyState: number = 0; // CONNECTING
  
  public onopen: (() => void) | null = null;
  public onmessage: ((event: any) => void) | null = null;
  public onerror: ((err: any) => void) | null = null;
  public onclose: ((event: any) => void) | null = null;

  public sentData: (string | ArrayBuffer | Blob)[] = [];
  public wasClosed = false;

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  public send(data: string | ArrayBuffer | Blob): void {
    this.sentData.push(data);
  }

  public close(code?: number, reason?: string): void {
    this.wasClosed = true;
    this.readyState = 3; // CLOSED
    if (this.onclose) {
      this.onclose({ code: code ?? 1000, reason: reason ?? '', wasClean: true });
    }
  }

  // Test triggers
  public triggerOpen(): void {
    this.readyState = 1; // OPEN
    if (this.onopen) {
      this.onopen();
    }
  }

  public triggerMessage(data: string): void {
    if (this.onmessage) {
      this.onmessage({ data });
    }
  }

  public triggerError(err: any): void {
    if (this.onerror) {
      this.onerror(err);
    }
  }
}

describe('ASRWebSocketClient', () => {
  beforeEach(() => {
    MockWebSocket.instances = [];
    vi.stubGlobal('WebSocket', MockWebSocket);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('should construct correctly and resolve ws url', () => {
    const callbacks: ASRStreamCallbacks = {};
    const client = new ASRWebSocketClient(callbacks, 'ws://custom-server/stream');
    
    client.connect({ method: 'greedy_search' });
    expect(MockWebSocket.instances.length).toBe(1);
    expect(MockWebSocket.instances[0].url).toBe('ws://custom-server/stream');
  });

  it('should send handshake config event on open', () => {
    const callbacks: ASRStreamCallbacks = {};
    const client = new ASRWebSocketClient(callbacks, 'ws://localhost/stream');

    client.connect({
      method: 'modified_beam_search',
      beam_size: 8,
      causal: true
    });

    const wsInstance = MockWebSocket.instances[0];
    expect(wsInstance.sentData.length).toBe(0);

    // Trigger open event
    wsInstance.triggerOpen();
    expect(wsInstance.sentData.length).toBe(1);
    
    const payload = JSON.parse(wsInstance.sentData[0] as string);
    expect(payload).toEqual({
      event: 'handshake',
      config: {
        method: 'modified_beam_search',
        beam_size: 8,
        causal: true
      }
    });
  });

  it('should invoke onHandshakeOk when receiving handshake_ok', () => {
    const onHandshakeOk = vi.fn();
    const callbacks: ASRStreamCallbacks = { onHandshakeOk };
    const client = new ASRWebSocketClient(callbacks, 'ws://localhost/stream');

    client.connect({ method: 'greedy_search' });
    const wsInstance = MockWebSocket.instances[0];
    wsInstance.triggerOpen();

    wsInstance.triggerMessage(JSON.stringify({
      event: 'handshake_ok',
      session_id: 'ws_session_xyz'
    }));

    expect(onHandshakeOk).toHaveBeenCalledWith('ws_session_xyz');
  });

  it('should invoke onTranscriptUpdate when receiving updates', () => {
    const onTranscriptUpdate = vi.fn();
    const callbacks: ASRStreamCallbacks = { onTranscriptUpdate };
    const client = new ASRWebSocketClient(callbacks, 'ws://localhost/stream');

    client.connect({ method: 'greedy_search' });
    const wsInstance = MockWebSocket.instances[0];
    wsInstance.triggerOpen();

    wsInstance.triggerMessage(JSON.stringify({
      event: 'transcript_update',
      text: 'chào các bạn',
      is_final: false,
      confidence: 0.94
    }));

    expect(onTranscriptUpdate).toHaveBeenCalledWith('chào các bạn', false, 0.94);
  });

  it('should invoke onFinished when receiving finished event', () => {
    const onFinished = vi.fn();
    const callbacks: ASRStreamCallbacks = { onFinished };
    const client = new ASRWebSocketClient(callbacks, 'ws://localhost/stream');

    client.connect({ method: 'greedy_search' });
    const wsInstance = MockWebSocket.instances[0];
    wsInstance.triggerOpen();

    wsInstance.triggerMessage(JSON.stringify({
      event: 'finished',
      full_transcript: 'chào các bạn.',
      confidence: 0.96
    }));

    expect(onFinished).toHaveBeenCalledWith('chào các bạn.', 0.96);
  });

  it('should allow sending binary audio chunks and stop event', () => {
    const callbacks: ASRStreamCallbacks = {};
    const client = new ASRWebSocketClient(callbacks, 'ws://localhost/stream');

    client.connect({ method: 'greedy_search' });
    const wsInstance = MockWebSocket.instances[0];
    wsInstance.triggerOpen();

    // Mock ws state to OPEN
    wsInstance.readyState = 1;

    const audioChunk = new ArrayBuffer(512);
    client.sendAudioChunk(audioChunk);
    expect(wsInstance.sentData[1]).toBe(audioChunk); // sentData[0] is handshake config

    client.sendStop();
    expect(JSON.parse(wsInstance.sentData[2] as string)).toEqual({ event: 'stop' });
  });

  it('should trigger onClose callbacks when connection closes', () => {
    const onClose = vi.fn();
    const callbacks: ASRStreamCallbacks = { onClose };
    const client = new ASRWebSocketClient(callbacks, 'ws://localhost/stream');

    client.connect({ method: 'greedy_search' });
    const wsInstance = MockWebSocket.instances[0];
    wsInstance.triggerOpen();

    wsInstance.close();
    expect(onClose).toHaveBeenCalledWith(1000, '');
  });
});
