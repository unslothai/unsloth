"use client";

import { useState, useEffect } from "react";
import { VoiceAgent } from "@/components/voice-agent";
import { AppointmentList } from "@/components/appointment-list";
import { Calendar, Phone } from "lucide-react";

interface Appointment {
  id: string;
  summary: string;
  start: string;
  end: string;
  description: string;
}

export default function Home() {
  const [appointments, setAppointments] = useState<Appointment[]>([]);
  const [isCallActive, setIsCallActive] = useState(false);

  // Fetch appointments on load
  useEffect(() => {
    fetchAppointments();
  }, []);

  // Refresh appointments after call ends
  useEffect(() => {
    if (!isCallActive) {
      fetchAppointments();
    }
  }, [isCallActive]);

  const fetchAppointments = async () => {
    try {
      const response = await fetch("/api/appointments");
      if (response.ok) {
        const data = await response.json();
        setAppointments(data.appointments || []);
      }
    } catch (error) {
      console.error("Error fetching appointments:", error);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-b from-background to-secondary/20">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <header className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-primary to-blue-400 bg-clip-text text-transparent">
            Ana
          </h1>
          <p className="text-muted-foreground">Asistente Médica Virtual</p>
        </header>

        {/* Main content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 max-w-6xl mx-auto">
          {/* Voice Agent Panel */}
          <div className="flex flex-col items-center">
            <VoiceAgent
              onCallStateChange={setIsCallActive}
            />
          </div>

          {/* Appointments Panel */}
          <div className="bg-card rounded-2xl p-6 border border-border">
            <div className="flex items-center gap-2 mb-6">
              <Calendar className="w-5 h-5 text-primary" />
              <h2 className="text-xl font-semibold">Próximas Citas</h2>
            </div>
            <AppointmentList appointments={appointments} />
          </div>
        </div>

        {/* Instructions */}
        <div className="mt-12 text-center text-muted-foreground">
          <div className="flex items-center justify-center gap-2 mb-2">
            <Phone className="w-4 h-4" />
            <span>Presiona el botón para llamar a Ana</span>
          </div>
          <p className="text-sm">
            Puedes pedir citas, cancelarlas o consultar tu agenda
          </p>
        </div>
      </div>
    </main>
  );
}
