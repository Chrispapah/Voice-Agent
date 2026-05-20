import { getOpenAIRealtimeVoiceSessionWebSocketUrl, resolveAccessToken } from "./api";
import type { VoiceSessionCallbacks } from "./voiceSession";

type ServerJson =
  | { type: "ready"; conversation_id: string }
  | { type: "transcript.partial"; text: string }
  | { type: "transcript.final"; text: string }
  | { type: "agent.text"; text: string }
  | { type: "agent.audio"; chunk: string; format?: "pcm16"; sample_rate?: number }
  | { type: "agent.audio_segment_end"; format?: "pcm16"; sample_rate?: number }
  | { type: "agent.done" }
  | { type: "agent.interrupted" }
  | { type: "error"; message: string }
  | { type: "pong" };

const REALTIME_INPUT_SAMPLE_RATE = 24_000;
const REALTIME_OUTPUT_SAMPLE_RATE = 24_000;

function decodeBase64Pcm16(base64: string): Int16Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return new Int16Array(bytes.buffer);
}

function downsampleToPcm16(input: Float32Array, inputSampleRate: number, outputSampleRate: number): Int16Array {
  if (outputSampleRate === inputSampleRate) {
    const output = new Int16Array(input.length);
    for (let i = 0; i < input.length; i += 1) {
      const s = Math.max(-1, Math.min(1, input[i] || 0));
      output[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
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
 * Browser voice client for the OpenAI Realtime addon.
 * Sends raw PCM16 mic frames to the backend and plays raw PCM16 audio deltas.
 */
export class OpenAIRealtimeVoiceSession {
  private ws: WebSocket | null = null;
  private stream: MediaStream | null = null;
  private audioContext: AudioContext | null = null;
  private processor: ScriptProcessorNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private mutedGain: GainNode | null = null;
  private playbackChain: Promise<void> = Promise.resolve();
  private readonly activeSources: AudioBufferSourceNode[] = [];
  private readonly callbacks: VoiceSessionCallbacks;

  constructor(callbacks: VoiceSessionCallbacks = {}) {
    this.callbacks = callbacks;
  }

  get isActive(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  async start(botId: string, leadId: string, conversationId?: string | null): Promise<void> {
    await this.stop();
    const token = await resolveAccessToken();
    const ws = new WebSocket(getOpenAIRealtimeVoiceSessionWebSocketUrl(botId));
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
          this.callbacks.onTranscriptFinal?.(msg.text);
          break;
        case "agent.text":
          this.callbacks.onAgentText?.(msg.text);
          break;
        case "agent.audio":
          this.queuePcmPlayback(decodeBase64Pcm16(msg.chunk), msg.sample_rate || REALTIME_OUTPUT_SAMPLE_RATE);
          break;
        case "agent.done":
          this.callbacks.onAgentDone?.();
          break;
        case "agent.interrupted":
          this.stopPlayback();
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
    return this.activeSources.length > 0;
  }

  private async ensureAudioContext(): Promise<AudioContext> {
    if (!this.audioContext) this.audioContext = new AudioContext({ latencyHint: "interactive" });
    if (this.audioContext.state === "suspended") {
      await this.audioContext.resume();
    }
    return this.audioContext;
  }

  private queuePcmPlayback(samples: Int16Array, sampleRate: number): void {
    const run = async (): Promise<void> => {
      if (samples.length === 0) return;
      const ctx = await this.ensureAudioContext();
      const buffer = ctx.createBuffer(1, samples.length, sampleRate);
      const channel = buffer.getChannelData(0);
      for (let i = 0; i < samples.length; i += 1) {
        channel[i] = Math.max(-1, Math.min(1, samples[i] / 0x8000));
      }
      await new Promise<void>((resolve) => {
        const src = ctx.createBufferSource();
        src.buffer = buffer;
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
      this.callbacks.onError?.(e instanceof Error ? e.message : "Microphone permission denied");
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
    this.playbackChain = Promise.resolve();
  }
}
