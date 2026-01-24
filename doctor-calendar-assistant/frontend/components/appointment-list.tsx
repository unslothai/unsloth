"use client";

import { Calendar, Clock, User } from "lucide-react";
import { cn } from "@/lib/utils";

interface Appointment {
  id: string;
  summary: string;
  start: string;
  end: string;
  description: string;
}

interface AppointmentListProps {
  appointments: Appointment[];
  className?: string;
}

export function AppointmentList({ appointments, className }: AppointmentListProps) {
  if (appointments.length === 0) {
    return (
      <div className={cn("text-center py-8 text-muted-foreground", className)}>
        <Calendar className="w-12 h-12 mx-auto mb-4 opacity-50" />
        <p>No hay citas próximas</p>
        <p className="text-sm mt-2">Llama a Ana para agendar una cita</p>
      </div>
    );
  }

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const days = ["Dom", "Lun", "Mar", "Mié", "Jue", "Vie", "Sáb"];
    const months = [
      "Ene", "Feb", "Mar", "Abr", "May", "Jun",
      "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"
    ];

    return {
      day: days[date.getDay()],
      date: date.getDate(),
      month: months[date.getMonth()],
      time: date.toLocaleTimeString("es-MX", {
        hour: "numeric",
        minute: "2-digit",
        hour12: true,
      }),
    };
  };

  const isToday = (dateString: string) => {
    const date = new Date(dateString);
    const today = new Date();
    return date.toDateString() === today.toDateString();
  };

  const isTomorrow = (dateString: string) => {
    const date = new Date(dateString);
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);
    return date.toDateString() === tomorrow.toDateString();
  };

  return (
    <div className={cn("space-y-3", className)}>
      {appointments.map((apt) => {
        const formatted = formatDate(apt.start);
        const today = isToday(apt.start);
        const tomorrow = isTomorrow(apt.start);

        return (
          <div
            key={apt.id}
            className={cn(
              "flex items-start gap-4 p-4 rounded-xl transition-colors",
              "bg-secondary/50 hover:bg-secondary/70",
              today && "border-l-4 border-primary"
            )}
          >
            {/* Date badge */}
            <div className="flex-shrink-0 text-center">
              <div
                className={cn(
                  "w-14 h-14 rounded-lg flex flex-col items-center justify-center",
                  today
                    ? "bg-primary text-primary-foreground"
                    : "bg-muted text-muted-foreground"
                )}
              >
                <span className="text-xs font-medium uppercase">
                  {today ? "Hoy" : tomorrow ? "Mañana" : formatted.day}
                </span>
                <span className="text-lg font-bold">{formatted.date}</span>
              </div>
            </div>

            {/* Appointment details */}
            <div className="flex-1 min-w-0">
              <h4 className="font-medium truncate">
                {apt.summary || "Cita médica"}
              </h4>

              <div className="flex items-center gap-4 mt-1 text-sm text-muted-foreground">
                <div className="flex items-center gap-1">
                  <Clock className="w-4 h-4" />
                  <span>{formatted.time}</span>
                </div>
                <div className="flex items-center gap-1">
                  <Calendar className="w-4 h-4" />
                  <span>{formatted.month}</span>
                </div>
              </div>

              {apt.description && (
                <p className="mt-2 text-sm text-muted-foreground truncate">
                  {apt.description}
                </p>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
