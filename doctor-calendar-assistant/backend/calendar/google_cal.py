"""
Google Calendar API integration
Handles authentication and calendar operations
"""

import asyncio
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# Required scopes for calendar access
SCOPES = ["https://www.googleapis.com/auth/calendar"]


class GoogleCalendarClient:
    """Client for Google Calendar API operations"""

    def __init__(self, config: dict):
        self.credentials_file = Path(
            config.get("credentials_file", "config/google_credentials.json")
        )
        self.token_file = Path(config.get("token_file", "config/google_token.json"))
        self.calendar_id = config.get("calendar_id", "primary")
        self.timezone = config.get("timezone", "America/Mexico_City")

        self.service = None
        self._authenticate()

    def _authenticate(self):
        """Authenticate with Google Calendar API"""
        creds = None

        # Load existing token
        if self.token_file.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_file), SCOPES)

        # Refresh or get new credentials
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not self.credentials_file.exists():
                    print(
                        f"    ⚠ Google credentials file not found: {self.credentials_file}"
                    )
                    print(
                        "    Please follow the setup instructions to configure Google Calendar"
                    )
                    self.service = None
                    return

                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self.credentials_file), SCOPES
                )
                creds = flow.run_local_server(port = 0)

            # Save credentials
            self.token_file.parent.mkdir(parents = True, exist_ok = True)
            with open(self.token_file, "w") as token:
                token.write(creds.to_json())

        # Build service
        self.service = build("calendar", "v3", credentials = creds)
        print("    ✓ Google Calendar connected")

    async def get_busy_times(self, date: str) -> list:
        """
        Get busy time slots for a specific date

        Args:
            date: Date in YYYY-MM-DD format

        Returns:
            List of busy time slots in HH:MM format
        """
        if not self.service:
            return []

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_busy_times_sync, date)

    def _get_busy_times_sync(self, date: str) -> list:
        """Synchronous version of get_busy_times"""
        try:
            # Parse date and create time bounds
            target_date = datetime.strptime(date, "%Y-%m-%d")
            time_min = target_date.replace(hour = 0, minute = 0, second = 0).isoformat() + "Z"
            time_max = (
                target_date.replace(hour = 23, minute = 59, second = 59).isoformat() + "Z"
            )

            # Query calendar
            events_result = (
                self.service.events()
                .list(
                    calendarId = self.calendar_id,
                    timeMin = time_min,
                    timeMax = time_max,
                    singleEvents = True,
                    orderBy = "startTime",
                )
                .execute()
            )

            events = events_result.get("items", [])

            # Extract busy times
            busy_times = []
            for event in events:
                start = event["start"].get("dateTime", event["start"].get("date"))
                if "T" in start:
                    # Has time component
                    event_time = datetime.fromisoformat(start.replace("Z", "+00:00"))
                    busy_times.append(event_time.strftime("%H:%M"))

            return busy_times

        except HttpError as e:
            print(f"Calendar API error: {e}")
            return []

    async def get_upcoming_appointments(
        self, max_results: int = 10, search_query: Optional[str] = None
    ) -> list:
        """
        Get upcoming appointments

        Args:
            max_results: Maximum number of results
            search_query: Optional search term

        Returns:
            List of appointment dictionaries
        """
        if not self.service:
            return []

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._get_upcoming_sync, max_results, search_query
        )

    def _get_upcoming_sync(self, max_results: int, search_query: Optional[str]) -> list:
        """Synchronous version of get_upcoming_appointments"""
        try:
            now = datetime.utcnow().isoformat() + "Z"

            kwargs = {
                "calendarId": self.calendar_id,
                "timeMin": now,
                "maxResults": max_results,
                "singleEvents": True,
                "orderBy": "startTime",
            }

            if search_query:
                kwargs["q"] = search_query

            events_result = self.service.events().list(**kwargs).execute()
            events = events_result.get("items", [])

            appointments = []
            for event in events:
                appointments.append(
                    {
                        "id": event["id"],
                        "summary": event.get("summary", "Sin título"),
                        "start": event["start"].get(
                            "dateTime", event["start"].get("date")
                        ),
                        "end": event["end"].get("dateTime", event["end"].get("date")),
                        "description": event.get("description", ""),
                    }
                )

            return appointments

        except HttpError as e:
            print(f"Calendar API error: {e}")
            return []

    async def create_appointment(
        self,
        date: str,
        time: str,
        title: str,
        description: str = "",
        duration_minutes: int = 30,
    ) -> dict:
        """
        Create a new appointment

        Args:
            date: Date in YYYY-MM-DD format
            time: Time in HH:MM format
            title: Appointment title
            description: Appointment description
            duration_minutes: Duration in minutes

        Returns:
            Result dictionary with success status and appointment ID
        """
        if not self.service:
            return {"success": False, "error": "Calendar not connected"}

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._create_appointment_sync,
            date,
            time,
            title,
            description,
            duration_minutes,
        )

    def _create_appointment_sync(
        self, date: str, time: str, title: str, description: str, duration_minutes: int
    ) -> dict:
        """Synchronous version of create_appointment"""
        try:
            # Parse start time
            start_dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
            end_dt = start_dt + timedelta(minutes = duration_minutes)

            event = {
                "summary": title,
                "description": description,
                "start": {
                    "dateTime": start_dt.isoformat(),
                    "timeZone": self.timezone,
                },
                "end": {
                    "dateTime": end_dt.isoformat(),
                    "timeZone": self.timezone,
                },
                "reminders": {
                    "useDefault": False,
                    "overrides": [
                        {"method": "popup", "minutes": 30},
                    ],
                },
            }

            created_event = (
                self.service.events()
                .insert(calendarId = self.calendar_id, body = event)
                .execute()
            )

            return {
                "success": True,
                "id": created_event["id"],
                "link": created_event.get("htmlLink", ""),
            }

        except HttpError as e:
            print(f"Calendar API error: {e}")
            return {"success": False, "error": str(e)}

    async def delete_appointment(self, appointment_id: str) -> dict:
        """
        Delete/cancel an appointment

        Args:
            appointment_id: ID of the appointment to delete

        Returns:
            Result dictionary with success status
        """
        if not self.service:
            return {"success": False, "error": "Calendar not connected"}

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._delete_appointment_sync, appointment_id
        )

    def _delete_appointment_sync(self, appointment_id: str) -> dict:
        """Synchronous version of delete_appointment"""
        try:
            self.service.events().delete(
                calendarId = self.calendar_id, eventId = appointment_id
            ).execute()

            return {"success": True}

        except HttpError as e:
            print(f"Calendar API error: {e}")
            return {"success": False, "error": str(e)}

    async def update_appointment(
        self,
        appointment_id: str,
        date: Optional[str] = None,
        time: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> dict:
        """
        Update an existing appointment

        Args:
            appointment_id: ID of the appointment
            date: New date (optional)
            time: New time (optional)
            title: New title (optional)
            description: New description (optional)

        Returns:
            Result dictionary with success status
        """
        if not self.service:
            return {"success": False, "error": "Calendar not connected"}

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._update_appointment_sync,
            appointment_id,
            date,
            time,
            title,
            description,
        )

    def _update_appointment_sync(
        self,
        appointment_id: str,
        date: Optional[str],
        time: Optional[str],
        title: Optional[str],
        description: Optional[str],
    ) -> dict:
        """Synchronous version of update_appointment"""
        try:
            # Get existing event
            event = (
                self.service.events()
                .get(calendarId = self.calendar_id, eventId = appointment_id)
                .execute()
            )

            # Update fields
            if title:
                event["summary"] = title
            if description:
                event["description"] = description
            if date and time:
                start_dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
                # Keep same duration
                old_start = datetime.fromisoformat(
                    event["start"]["dateTime"].replace("Z", "+00:00")
                )
                old_end = datetime.fromisoformat(
                    event["end"]["dateTime"].replace("Z", "+00:00")
                )
                duration = old_end - old_start
                end_dt = start_dt + duration

                event["start"] = {
                    "dateTime": start_dt.isoformat(),
                    "timeZone": self.timezone,
                }
                event["end"] = {
                    "dateTime": end_dt.isoformat(),
                    "timeZone": self.timezone,
                }

            updated_event = (
                self.service.events()
                .update(calendarId = self.calendar_id, eventId = appointment_id, body = event)
                .execute()
            )

            return {"success": True, "id": updated_event["id"]}

        except HttpError as e:
            print(f"Calendar API error: {e}")
            return {"success": False, "error": str(e)}
