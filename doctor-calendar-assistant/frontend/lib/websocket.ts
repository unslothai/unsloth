/**
 * WebSocket client for real-time voice communication
 */

export type MessageType =
  | "status"
  | "transcript"
  | "response"
  | "audio"
  | "error";

export interface WebSocketMessage {
  type: MessageType;
  status?: string;
  message?: string;
  text?: string;
  role?: "user" | "assistant";
  data?: string; // Base64 audio data
}

export type MessageHandler = (message: WebSocketMessage) => void;

export class VoiceWebSocket {
  private ws: WebSocket | null = null;
  private clientId: string;
  private messageHandlers: MessageHandler[] = [];
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private baseUrl: string;

  constructor(baseUrl: string = "ws://localhost:8000") {
    this.baseUrl = baseUrl;
    this.clientId = this.generateClientId();
  }

  private generateClientId(): string {
    return `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(`${this.baseUrl}/ws/${this.clientId}`);

        this.ws.onopen = () => {
          console.log("WebSocket connected");
          this.reconnectAttempts = 0;
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data);
            this.messageHandlers.forEach((handler) => handler(message));
          } catch (error) {
            console.error("Error parsing message:", error);
          }
        };

        this.ws.onerror = (error) => {
          console.error("WebSocket error:", error);
          reject(error);
        };

        this.ws.onclose = () => {
          console.log("WebSocket closed");
          this.attemptReconnect();
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = Math.pow(2, this.reconnectAttempts) * 1000;
      console.log(`Attempting reconnect in ${delay}ms...`);
      setTimeout(() => this.connect(), delay);
    }
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  onMessage(handler: MessageHandler): void {
    this.messageHandlers.push(handler);
  }

  offMessage(handler: MessageHandler): void {
    this.messageHandlers = this.messageHandlers.filter((h) => h !== handler);
  }

  send(message: object): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }

  startCall(): void {
    this.send({ type: "start_call" });
  }

  endCall(): void {
    this.send({ type: "end_call" });
  }

  sendAudioChunk(data: string): void {
    this.send({ type: "audio_chunk", data });
  }

  sendAudioComplete(): void {
    this.send({ type: "audio_complete" });
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}
