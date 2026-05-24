import { getOpenAIRealtimeElevenLabsVoiceSessionWebSocketUrl, resolveAccessToken } from "./api";
import { formatMicrophoneError } from "./micErrors";
import type { VoiceSessionCallbacks, VoiceSessionStartOptions } from "./voiceSession";

type ServerJson =
  | { type: "ready"; conversation_id: string; allow_interruptions?: boolean }
  | { type: "transcript.partial"; text: string }
  | { type: "transcript.final"; text: string }
  | { type: "agent.text"; text: string }
  | { type: "agent.audio"; chunk: string }
  | { type: "agent.audio_segment_end" }
  | { type: "agent.done" }
  | { type: "agent.interrupted" }
  | { type: "error"; message: string }
  | { type: "pong" };

const REALTIME_INPUT_SAMPLE_RATE = 24_000;

function mergeChunks(chunks: Uint8Array[]): Uint8Array {
  const total = chunks.reduce((n, c) => n + c.length, 0);
  const merged = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

function downsampleToPcm16(input: Float32Array, inputSampleRate: number, outputSampleRate: number): Int16Array {
  if (outputSampleRate === inputSampleRate) {
    const output = new Int16Array(input.length);
    for (let i = 0; i < input.length; i += 1) {
      const sample = Math.max(-1, Math.min(1, input[i] || 0));
      output[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    }
    return output;
  }

  const ratio = inputSampleRate / outputSampleRate;
  const outputLength = Math.max(1, Math.round(input.length / ratio));
  const output = new Int16Array(outputLength);
  for (let i = 0; i < outputLength; i += 1) {
    const start = Math.floor(i * ratio);
    const end = Math.min(input.length, Math.floor((i + 1) * ratio));
    let sum = 0;
    let count = 0;
    for (let j = start; j < end; j += 1) {
      sum += input[j] || 0;
      count += 1;
    }
    const sample = Math.max(-1, Math.min(1, count > 0 ? sum / count : 0));
    output[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  return output;
}

/**
 * Hybrid browser voice client: PCM mic frames up, ElevenLabs MP3 segments down.
 */
export class OpenAIRealtimeElevenLabsVoiceSession {
  private ws: WebSocket | null = null;
  private stream: MediaStream | null = null;
  private audioContext: AudioContext | null = null;
  private processor: ScriptProcessorNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private mutedGain: GainNode | null = null;
  private phraseChunks: Uint8Array[] = [];
  private playbackChain: Promise<void> = Promise.resolve();
  private queuedPlaybackSegments = 0;
  private playbackGeneration = 0;
  private allowInterruptions = true;
  private readonly activeSources: AudioBufferSourceNode[] = [];
  private readonly callbacks: VoiceSessionCallbacks;

  constructor(callbacks: VoiceSessionCallbacks = {}) {
    this.callbacks = callbacks;
  }

  get isActive(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  async start(botId: string, leadId: string, conversationId?: string | null): Promise<void> {
    const token = await resolveAccessToken();
    await this.startWithOptions({
      wsUrl: getOpenAIRealtimeElevenLabsVoiceSessionWebSocketUrl(botId),
      authMessage: { type: "auth", access_token: token },
      leadId,
      conversationId,
    });
  }

  async startWithOptions(options: VoiceSessionStartOptions): Promise<void> {
    await this.stop();
    const ws = new WebSocket(options.wsUrl);
    this.ws = ws;

    await new Promise<void>((resolve, reject) => {
      const t = window.setTimeout(() => reject(new Error("WebSocket connect timeout")), 15000);
      ws.onopen = () => {
        window.clearTimeout(t);
        if (options.authMessage) {
          ws.send(JSON.stringify(options.authMessage));
        }
        ws.send(
          JSON.stringify({
            type: "session.start",
            lead_id: options.leadId,
            conversation_id: options.conversationId ?? null,
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
          this.allowInterruptions = msg.allow_interruptions ?? true;
          this.callbacks.onReady?.(msg.conversation_id);
          void this.startMic();
          break;
        case "transcript.partial":
          if (this.allowInterruptions && this.isPlaybackActive() && this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: "interrupt" }));
            this.stopPlayback();
          }
          this.callbacks.onTranscriptPartial?.(msg.text);
          break;
        case "transcript.final":
          if (this.allowInterruptions) {
            this.stopPlayback();
            this.phraseChunks = [];
          }
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

  private isPlaybackActive(): boolean {
    return this.activeSources.length > 0 || this.phraseChunks.length > 0 || this.queuedPlaybackSegments > 0;
  }

  private async ensureAudioContext(): Promise<AudioContext> {
    if (!this.audioContext) this.audioContext = new AudioContext({ latencyHint: "interactive" });
    if (this.audioContext.state === "suspended") {
      await this.audioContext.resume();
    }
    return this.audioContext;
  }

  private flushPhrasePlayback(): void {
    if (this.phraseChunks.length === 0) return;
    const merged = mergeChunks(this.phraseChunks);
    this.phraseChunks = [];
    this.queuedPlaybackSegments += 1;
    const playbackGeneration = this.playbackGeneration;
    const run = async (): Promise<void> => {
      try {
        if (merged.length === 0) return;
        if (playbackGeneration !== this.playbackGeneration) return;
        const ctx = await this.ensureAudioContext();
        if (playbackGeneration !== this.playbackGeneration) return;
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
      } finally {
        this.queuedPlaybackSegments = Math.max(0, this.queuedPlaybackSegments - 1);
      }
    };
    this.playbackChain = this.playbackChain.then(run).catch(() => undefined);
  }

  private async startMic(): Promise<void> {
    try {
      const ctx = await this.ensureAudioContext();
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      this.source = ctx.createMediaStreamSource(this.stream);
      this.processor = ctx.createScriptProcessor(4096, 1, 1);
      this.mutedGain = ctx.createGain();
      this.mutedGain.gain.value = 0;
      this.processor.onaudioprocess = (ev) => {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        const input = ev.inputBuffer.getChannelData(0);
        const pcm = downsampleToPcm16(input, ctx.sampleRate, REALTIME_INPUT_SAMPLE_RATE);
        const frame = pcm.buffer.slice(pcm.byteOffset, pcm.byteOffset + pcm.byteLength);
        this.ws.send(frame);
      };
      this.source.connect(this.processor);
      this.processor.connect(this.mutedGain);
      this.mutedGain.connect(ctx.destination);
    } catch (e) {
      this.callbacks.onError?.(formatMicrophoneError(e));
    }
  }

  private async stopMicOnly(): Promise<void> {
    if (this.processor) {
      this.processor.onaudioprocess = null;
      this.processor.disconnect();
    }
    this.processor = null;
    this.source?.disconnect();
    this.source = null;
    this.mutedGain?.disconnect();
    this.mutedGain = null;
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
  }

  private stopPlayback(): void {
    for (const source of [...this.activeSources]) {
      try {
        source.stop(0);
      } catch {
        /* noop */
      }
      try {
        source.disconnect();
      } catch {
        /* noop */
      }
    }
    this.activeSources.length = 0;
    this.queuedPlaybackSegments = 0;
    this.playbackGeneration += 1;
    this.playbackChain = Promise.resolve();
    this.phraseChunks = [];
  }
}
