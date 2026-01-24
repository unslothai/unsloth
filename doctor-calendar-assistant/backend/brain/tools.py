"""
Calendar tools for the assistant
Handles appointment scheduling, cancellation, and queries
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional
import json


class CalendarTools:
    """Tools for interacting with Google Calendar"""

    def __init__(self, calendar_client):
        self.calendar = calendar_client

    async def execute(self, function_name: str, arguments: dict) -> str:
        """
        Execute a tool function

        Args:
            function_name: Name of the function to execute
            arguments: Function arguments

        Returns:
            Result as a string for the LLM
        """
        tools = {
            "get_available_slots": self.get_available_slots,
            "book_appointment": self.book_appointment,
            "cancel_appointment": self.cancel_appointment,
            "list_appointments": self.list_appointments,
        }

        if function_name not in tools:
            return f"Error: Función '{function_name}' no encontrada"

        try:
            result = await tools[function_name](**arguments)
            return result
        except Exception as e:
            return f"Error al ejecutar {function_name}: {str(e)}"

    async def get_available_slots(self, date: str) -> str:
        """
        Get available appointment slots for a date

        Args:
            date: Date in YYYY-MM-DD format

        Returns:
            Available time slots as formatted string
        """
        try:
            # Parse date
            target_date = datetime.strptime(date, "%Y-%m-%d")

            # Get busy times from calendar
            busy_times = await self.calendar.get_busy_times(date)

            # Define available hours (9 AM to 5 PM)
            available_hours = [
                "09:00", "10:00", "11:00", "12:00",
                "14:00", "15:00", "16:00", "17:00"
            ]

            # Filter out busy times
            free_slots = []
            for slot in available_hours:
                if slot not in busy_times:
                    # Convert to friendly format
                    hour = int(slot.split(":")[0])
                    if hour < 12:
                        friendly = f"{hour} de la mañana"
                    elif hour == 12:
                        friendly = "12 del mediodía"
                    else:
                        friendly = f"{hour - 12} de la tarde"
                    free_slots.append({"time": slot, "friendly": friendly})

            if not free_slots:
                return f"No hay horarios disponibles para el {self._format_date(target_date)}"

            slots_text = ", ".join([s["friendly"] for s in free_slots[:4]])
            return f"Horarios disponibles para el {self._format_date(target_date)}: {slots_text}"

        except ValueError:
            return "Error: Formato de fecha inválido. Usa YYYY-MM-DD"

    async def book_appointment(
        self,
        date: str,
        time: str,
        patient_name: str,
        reason: str = "Consulta general"
    ) -> str:
        """
        Book an appointment

        Args:
            date: Date in YYYY-MM-DD format
            time: Time in HH:MM format
            patient_name: Patient's name
            reason: Reason for appointment

        Returns:
            Confirmation message
        """
        try:
            # Create appointment
            result = await self.calendar.create_appointment(
                date=date,
                time=time,
                title=f"Cita: {patient_name}",
                description=f"Paciente: {patient_name}\nMotivo: {reason}"
            )

            if result.get("success"):
                # Format confirmation
                target_date = datetime.strptime(date, "%Y-%m-%d")
                hour = int(time.split(":")[0])
                if hour < 12:
                    time_friendly = f"{hour} de la mañana"
                elif hour == 12:
                    time_friendly = "12 del mediodía"
                else:
                    time_friendly = f"{hour - 12} de la tarde"

                return (
                    f"Cita agendada exitosamente para {patient_name} "
                    f"el {self._format_date(target_date)} a las {time_friendly}. "
                    f"ID de confirmación: {result.get('id', 'N/A')}"
                )
            else:
                return f"No se pudo agendar la cita: {result.get('error', 'Error desconocido')}"

        except Exception as e:
            return f"Error al agendar cita: {str(e)}"

    async def cancel_appointment(self, appointment_id: str) -> str:
        """
        Cancel an existing appointment

        Args:
            appointment_id: ID of the appointment to cancel

        Returns:
            Confirmation message
        """
        try:
            result = await self.calendar.delete_appointment(appointment_id)

            if result.get("success"):
                return "La cita ha sido cancelada exitosamente."
            else:
                return f"No se pudo cancelar la cita: {result.get('error', 'Error desconocido')}"

        except Exception as e:
            return f"Error al cancelar cita: {str(e)}"

    async def list_appointments(self, patient_name: Optional[str] = None) -> str:
        """
        List upcoming appointments

        Args:
            patient_name: Optional filter by patient name

        Returns:
            List of appointments as formatted string
        """
        try:
            appointments = await self.calendar.get_upcoming_appointments(
                max_results=5,
                search_query=patient_name
            )

            if not appointments:
                if patient_name:
                    return f"No encontré citas para {patient_name}"
                return "No hay citas próximas agendadas"

            # Format appointments
            apt_list = []
            for apt in appointments:
                date = datetime.fromisoformat(apt["start"].replace("Z", "+00:00"))
                apt_list.append(
                    f"- {self._format_date(date)} a las {date.strftime('%I:%M %p')}: "
                    f"{apt.get('summary', 'Sin descripción')}"
                )

            return "Próximas citas:\n" + "\n".join(apt_list)

        except Exception as e:
            return f"Error al obtener citas: {str(e)}"

    def _format_date(self, date: datetime) -> str:
        """Format date in friendly Spanish"""
        days = [
            "lunes", "martes", "miércoles", "jueves",
            "viernes", "sábado", "domingo"
        ]
        months = [
            "enero", "febrero", "marzo", "abril", "mayo", "junio",
            "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"
        ]

        day_name = days[date.weekday()]
        month_name = months[date.month - 1]

        return f"{day_name} {date.day} de {month_name}"
