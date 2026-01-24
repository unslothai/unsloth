"use client";

import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Phone, PhoneOff, Mic, MicOff } from "lucide-react";
import { Orb, type AgentState } from "./ui/orb";
import { Waveform } from "./ui/waveform";
import { VoiceWebSocket, WebSocketMessage } from "@/lib/websocket";
import { AudioRecorder, AudioPlayer, base64ToArrayBuffer, arrayBufferToBase64 } from "@/lib/audio";
import { cn } from "@/lib/utils";

interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  timestamp: Date;
}

interface VoiceAgentProps {
  onCallStateChange?: (isActive: boolean) => void;
}

export function VoiceAgent({ onCallStateChange }: VoiceAgentProps) {
  const [isCallActive, setIsCallActive] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [status, setStatus] = useState<string>("Desconectado");
  const [messages, setMessages] = useState<Message[]>([]);
  const [callDuration, setCallDuration] = useState(0);

  const wsRef = useRef<VoiceWebSocket | null>(null);
  const recorderRef = useRef<AudioRecorder | null>(null);
  const playerRef = useRef<AudioPlayer | null>(null);
  const callTimerRef = useRef<NodeJS.Timeout | null>(null);
  const recordingTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Initialize audio player
  useEffect(() => {
    playerRef.current = new AudioPlayer();
    return () => {
      playerRef.current?.stop();
    };
  }, []);

  // Call duration timer
  useEffect(() => {
    if (isCallActive) {
      callTimerRef.current = setInterval(() => {
        setCallDuration((d) => d + 1);
      }, 1000);
    } else {
      if (callTimerRef.current) {
        clearInterval(callTimerRef.current);
      }
      setCallDuration(0);
    }
    return () => {
      if (callTimerRef.current) {
        clearInterval(callTimerRef.current);
      }
    };
  }, [isCallActive]);

  // Notify parent of call state changes
  useEffect(() => {
    onCallStateChange?.(isCallActive);
  }, [isCallActive, onCallStateChange]);

  const handleMessage = useCallback((message: WebSocketMessage) => {
    switch (message.type) {
      case "status":
        setStatus(message.message || message.status || "");
        if (message.status === "listening") {
          setIsListening(true);
          setIsSpeaking(false);
          startRecording();
        } else if (message.status === "speaking") {
          setIsSpeaking(true);
          setIsListening(false);
          stopRecording();
        } else if (message.status === "thinking" || message.status === "processing") {
          setIsListening(false);
          setIsSpeaking(false);
          stopRecording();
        } else if (message.status === "call_ended") {
          setIsCallActive(false);
          setIsListening(false);
          setIsSpeaking(false);
        }
        break;

      case "transcript":
        if (message.text) {
          addMessage("user", message.text);
        }
        break;

      case "response":
        if (message.text) {
          addMessage("assistant", message.text);
        }
        break;

      case "audio":
        if (message.data) {
          const audioData = base64ToArrayBuffer(message.data);
          playerRef.current?.queueAudio(audioData);
        }
        break;

      case "error":
        console.error("Server error:", message.message);
        setStatus(`Error: ${message.message}`);
        break;
    }
  }, []);

  const addMessage = (role: "user" | "assistant", text: string) => {
    setMessages((prev) => [
      ...prev,
      {
        id: `${Date.now()}-${Math.random()}`,
        role,
        text,
        timestamp: new Date(),
      },
    ]);
  };

  const startRecording = async () => {
    if (isMuted || !wsRef.current) return;

    if (!recorderRef.current) {
      recorderRef.current = new AudioRecorder();
    }

    const started = await recorderRef.current.start((data) => {
      // Send audio chunk to server
      const base64 = arrayBufferToBase64(data);
      wsRef.current?.sendAudioChunk(base64);
    });

    if (started) {
      // Set timeout to stop recording after silence
      recordingTimeoutRef.current = setTimeout(() => {
        stopRecording();
        wsRef.current?.sendAudioComplete();
      }, 5000); // 5 second timeout
    }
  };

  const stopRecording = () => {
    if (recordingTimeoutRef.current) {
      clearTimeout(recordingTimeoutRef.current);
    }
    recorderRef.current?.stop();
  };

  const startCall = async () => {
    try {
      // Connect WebSocket
      wsRef.current = new VoiceWebSocket("ws://localhost:8000");
      wsRef.current.onMessage(handleMessage);
      await wsRef.current.connect();

      setIsConnected(true);
      setIsCallActive(true);
      setMessages([]);
      setStatus("Conectando...");

      // Start the call
      wsRef.current.startCall();
    } catch (error) {
      console.error("Error starting call:", error);
      setStatus("Error de conexión");
      setIsCallActive(false);
    }
  };

  const endCall = () => {
    wsRef.current?.endCall();
    stopRecording();
    playerRef.current?.stop();

    setTimeout(() => {
      wsRef.current?.disconnect();
      wsRef.current = null;
      setIsConnected(false);
      setIsCallActive(false);
      setIsListening(false);
      setIsSpeaking(false);
      setStatus("Desconectado");
    }, 2000); // Wait for farewell message
  };

  const toggleMute = () => {
    setIsMuted(!isMuted);
    if (!isMuted) {
      stopRecording();
    }
  };

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  // Determine agent state for the Orb component (ElevenLabs UI style)
  const agentState: AgentState = useMemo(() => {
    if (!isCallActive) return "idle";
    if (isSpeaking) return "speaking";
    if (isListening) return "listening";
    if (status === "Pensando..." || status === "Procesando...") return "thinking";
    return "idle";
  }, [isCallActive, isSpeaking, isListening, status]);

  return (
    <div className="flex flex-col items-center w-full max-w-md">
      {/* Call panel */}
      <motion.div
        className={cn(
          "w-full rounded-2xl p-8 transition-all duration-500",
          isCallActive
            ? "bg-gradient-to-b from-card to-secondary/50 border border-primary/30"
            : "bg-card border border-border"
        )}
        layout
      >
        {/* Orb - ElevenLabs UI style with Three.js */}
        <div className="flex justify-center mb-6">
          <Orb
            state={agentState}
            colors={["#3b82f6", "#8b5cf6"]}
            size={isCallActive ? 200 : 150}
            volume={isListening || isSpeaking ? 0.5 : 0}
          />
        </div>

        {/* Agent name and status */}
        <div className="text-center mb-6">
          <h3 className="text-2xl font-semibold mb-1">Ana</h3>
          <p className="text-muted-foreground text-sm">
            {isCallActive ? status : "Asistente Médica"}
          </p>
          {isCallActive && (
            <p className="text-primary mt-2 font-mono">
              {formatDuration(callDuration)}
            </p>
          )}
        </div>

        {/* Waveform */}
        <AnimatePresence>
          {isCallActive && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="mb-6"
            >
              <Waveform isActive={isListening || isSpeaking} />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Messages */}
        <AnimatePresence>
          {isCallActive && messages.length > 0 && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="mb-6 max-h-48 overflow-y-auto rounded-lg bg-background/50 p-4"
            >
              {messages.slice(-4).map((msg) => (
                <div
                  key={msg.id}
                  className={cn(
                    "mb-2 last:mb-0",
                    msg.role === "user" ? "text-right" : "text-left"
                  )}
                >
                  <span
                    className={cn(
                      "inline-block px-3 py-2 rounded-lg text-sm",
                      msg.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : "bg-secondary text-secondary-foreground"
                    )}
                  >
                    {msg.text}
                  </span>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Call controls */}
        <div className="flex justify-center gap-4">
          {isCallActive ? (
            <>
              {/* Mute button */}
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={toggleMute}
                className={cn(
                  "p-4 rounded-full transition-colors",
                  isMuted
                    ? "bg-destructive text-destructive-foreground"
                    : "bg-secondary text-secondary-foreground hover:bg-secondary/80"
                )}
              >
                {isMuted ? <MicOff size={24} /> : <Mic size={24} />}
              </motion.button>

              {/* End call button */}
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={endCall}
                className="p-4 rounded-full bg-destructive text-destructive-foreground hover:bg-destructive/90"
              >
                <PhoneOff size={24} />
              </motion.button>
            </>
          ) : (
            /* Start call button */
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={startCall}
              className="flex items-center gap-3 px-8 py-4 rounded-full bg-primary text-primary-foreground hover:bg-primary/90 font-medium"
            >
              <Phone size={24} />
              <span>Llamar a Ana</span>
            </motion.button>
          )}
        </div>
      </motion.div>
    </div>
  );
}
