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
  | { type: "agent.done" }
  | { type: "agent.interrupted" }
  | { type: "error"; message: string }
  | { type: "pong" };

/**
 * Browser realtime voice client: WebSocket to backend (Deepgram STT + ElevenLabs TTS).
 */
export class VoiceSession {
  private ws: WebSocket | null = null;
  private recorder: MediaRecorder | null = null;
  private stream: MediaStream | null = null;
  private audioChunks: Uint8Array[] = [];
  private playingAudio: HTMLAudioElement | null = null;
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
          if (this.playingAudio && this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: "interrupt" }));
            this.stopPlayback();
          }
          this.callbacks.onTranscriptPartial?.(msg.text);
          break;
        case "transcript.final":
          this.stopPlayback();
          this.audioChunks = [];
          this.callbacks.onTranscriptFinal?.(msg.text);
          break;
        case "agent.text":
          this.audioChunks = [];
          this.callbacks.onAgentText?.(msg.text);
          break;
        case "agent.audio": {
          const binary = atob(msg.chunk);
          const buf = new Uint8Array(binary.length);
          for (let i = 0; i < binary.length; i += 1) buf[i] = binary.charCodeAt(i);
          this.audioChunks.push(buf);
          break;
        }
        case "agent.done":
          void this.playPendingAudio();
          this.callbacks.onAgentDone?.();
          break;
        case "agent.interrupted":
          this.stopPlayback();
          this.audioChunks = [];
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
    if (this.playingAudio) {
      this.playingAudio.pause();
      this.playingAudio.src = "";
      this.playingAudio = null;
    }
  }

  private async playPendingAudio(): Promise<void> {
    if (this.audioChunks.length === 0) return;
    const total = this.audioChunks.reduce((n, c) => n + c.length, 0);
    const merged = new Uint8Array(total);
    let o = 0;
    for (const c of this.audioChunks) {
      merged.set(c, o);
      o += c.length;
    }
    this.audioChunks = [];
    const blob = new Blob([merged], { type: "audio/mpeg" });
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    this.playingAudio = audio;
    audio.onended = () => {
      URL.revokeObjectURL(url);
      if (this.playingAudio === audio) this.playingAudio = null;
    };
    try {
      await audio.play();
    } catch {
      URL.revokeObjectURL(url);
    }
  }
}
