import { getVoiceSessionWebSocketUrl, resolveAccessToken } from "./api";

export type VoiceSessionCallbacks = {
  onReady?: (conversationId: string) => void;
  onTranscriptPartial?: (text: string) => void;
  onTranscriptFinal?: (text: string) => void;
  onAgentText?: (text: string) => void;
  onAgentDone?: () => void;
  onInterrupted?: () => void;
  onError?: (message: string) => void;
  /** Fired when the socket closes (including after {@link VoiceSession.stop}). */
  onClose?: () => void;
};

type ServerJson =
  | { type: "ready"; conversation_id: string }
  | { type: "transcript.partial"; text: string }
  | { type: "transcript.final"; text: string }
  | { type: "agent.text"; text: string }
  | { type: "agent.audio"; chunk: string }
  /** End of one TTS synthesis (typically one streamed LLM phrase). Client decodes queued MP3 and plays sequentially. */
  | { type: "agent.audio_segment_end" }
  | { type: "agent.done" }
  | { type: "agent.interrupted" }
  | { type: "error"; message: string }
  | { type: "pong" };

function mergeChunks(chunks: Uint8Array[]): Uint8Array {
  const total = chunks.reduce((n, c) => n + c.length, 0);
  const merged = new Uint8Array(total);
  let o = 0;
  for (const c of chunks) {
    merged.set(c, o);
    o += c.length;
  }
  return merged;
}

/**
 * Browser realtime voice client: WebSocket to backend (Deepgram STT + ElevenLabs TTS).
 * Plays TTS phrase-by-phrase via Web Audio (decode AudioBuffer per segment).
 */
export class VoiceSession {
  private ws: WebSocket | null = null;
  private recorder: MediaRecorder | null = null;
  private stream: MediaStream | null = null;
  /** MP3 fragments for the current synthesis segment */
  private phraseChunks: Uint8Array[] = [];
  private playbackChain: Promise<void> = Promise.resolve();
  private audioContext: AudioContext | null = null;
  private readonly activeSources: AudioBufferSourceNode[] = [];
  private readonly callbacks: VoiceSessionCallbacks;

  constructor(callbacks: VoiceSessionCallbacks = {}) {
    this.callbacks = callbacks;
  }

  get isActive(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  /** True while agent speech is playing or queued (phrase chain). */
  private isPlaybackActive(): boolean {
    return this.activeSources.length > 0 || this.phraseChunks.length > 0;
  }

  private async ensureAudioContext(): Promise<AudioContext> {
    if (!this.audioContext) this.audioContext = new AudioContext();
    if (this.audioContext.state === "suspended") {
      await this.audioContext.resume();
    }
    return this.audioContext;
  }

  /** Queue one complete MP3 segment (concat of chunks since last flush). */
  private flushPhrasePlayback(): void {
    if (this.phraseChunks.length === 0) return;
    const merged = mergeChunks(this.phraseChunks);
    this.phraseChunks = [];
    const run = async (): Promise<void> => {
      if (merged.length === 0) return;
      const ctx = await this.ensureAudioContext();
      const copy = new Uint8Array(merged.byteLength);
      copy.set(merged);
      let audioBuf: AudioBuffer;
      try {
        const ab = new ArrayBuffer(copy.byteLength);
        new Uint8Array(ab).set(copy);
        audioBuf = await ctx.decodeAudioData(ab);
      } catch {
        return;
      }
      await new Promise<void>((resolve) => {
        const src = ctx.createBufferSource();
        src.buffer = audioBuf;
        src.connect(ctx.destination);
        this.activeSources.push(src);
        src.onended = () => {
          const idx = this.activeSources.indexOf(src);
          if (idx >= 0) this.activeSources.splice(idx, 1);
          resolve();
        };
        try {
          src.start();
        } catch {
          const idx = this.activeSources.indexOf(src);
          if (idx >= 0) this.activeSources.splice(idx, 1);
          resolve();
        }
      });
    };
    this.playbackChain = this.playbackChain.then(run).catch(() => undefined);
  }

  async start(botId: string, leadId: string, conversationId?: string | null): Promise<void> {
    await this.stop();
    const token = await resolveAccessToken();
    const url = getVoiceSessionWebSocketUrl(botId);
    const ws = new WebSocket(url);
    this.ws = ws;

    await new Promise<void>((resolve, reject) => {
      const t = window.setTimeout(() => reject(new Error("WebSocket connect timeout")), 15000);
      ws.onopen = () => {
        window.clearTimeout(t);
        ws.send(JSON.stringify({ type: "auth", access_token: token }));
        ws.send(
          JSON.stringify({
            type: "session.start",
            lead_id: leadId,
            conversation_id: conversationId ?? null,
          }),
        );
        resolve();
      };
      ws.onerror = () => {
        window.clearTimeout(t);
        reject(new Error("WebSocket connection failed"));
      };
    });

    ws.onmessage = (ev) => {
      if (typeof ev.data !== "string") return;
      let msg: ServerJson;
      try {
        msg = JSON.parse(ev.data) as ServerJson;
      } catch {
        return;
      }
      switch (msg.type) {
        case "ready":
          this.callbacks.onReady?.(msg.conversation_id);
          void this.startMic();
          break;
        case "transcript.partial":
          if (this.isPlaybackActive() && this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: "interrupt" }));
            this.stopPlayback();
          }
          this.callbacks.onTranscriptPartial?.(msg.text);
          break;
        case "transcript.final":
          this.stopPlayback();
          this.phraseChunks = [];
          this.callbacks.onTranscriptFinal?.(msg.text);
          break;
        case "agent.text":
          this.callbacks.onAgentText?.(msg.text);
          break;
        case "agent.audio": {
          const binary = atob(msg.chunk);
          const buf = new Uint8Array(binary.length);
          for (let i = 0; i < binary.length; i += 1) buf[i] = binary.charCodeAt(i);
          this.phraseChunks.push(buf);
          break;
        }
        case "agent.audio_segment_end":
          this.flushPhrasePlayback();
          break;
        case "agent.done":
          this.flushPhrasePlayback();
          this.callbacks.onAgentDone?.();
          break;
        case "agent.interrupted":
          this.stopPlayback();
          this.phraseChunks = [];
          this.callbacks.onInterrupted?.();
          break;
        case "error":
          this.callbacks.onError?.(msg.message);
          break;
        default:
          break;
      }
    };

    ws.onclose = () => {
      void this.stopMicOnly();
      this.callbacks.onClose?.();
    };
  }

  interrupt(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "interrupt" }));
    }
    this.stopPlayback();
  }

  async stop(): Promise<void> {
    this.stopPlayback();
    await this.stopMicOnly();
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    if (this.audioContext?.state !== "closed") {
      await this.audioContext?.close().catch(() => undefined);
    }
    this.audioContext = null;
  }

  private async startMic(): Promise<void> {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/webm")
          ? "audio/webm"
          : "";
      this.recorder = mime
        ? new MediaRecorder(this.stream, { mimeType: mime })
        : new MediaRecorder(this.stream);
      this.recorder.ondataavailable = (ev) => {
        if (ev.data.size > 0 && this.ws && this.ws.readyState === WebSocket.OPEN) {
          void ev.data.arrayBuffer().then((buf) => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
              this.ws.send(buf);
            }
          });
        }
      };
      this.recorder.start(250);
    } catch (e) {
      this.callbacks.onError?.(e instanceof Error ? e.message : "Microphone permission denied");
    }
  }

  private async stopMicOnly(): Promise<void> {
    if (this.recorder && this.recorder.state !== "inactive") {
      this.recorder.stop();
    }
    this.recorder = null;
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
  }

  private stopPlayback(): void {
    for (const s of [...this.activeSources]) {
      try {
        s.stop(0);
      } catch {
        /* noop */
      }
      try {
        s.disconnect();
      } catch {
        /* noop */
      }
    }
    this.activeSources.length = 0;
    this.playbackChain = Promise.resolve();
    this.phraseChunks = [];
  }
}
