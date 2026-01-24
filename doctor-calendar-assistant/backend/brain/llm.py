"""
PersonaPlex LLM integration via Ollama
Handles conversation and tool calling for the assistant
"""

import asyncio
import json
import re
from typing import Optional

import httpx

from .prompts import get_full_system_prompt, get_conversation_context
from .tools import CalendarTools


class PersonaPlexBrain:
    """PersonaPlex-7B brain for the assistant via Ollama"""

    def __init__(self, llm_config: dict, assistant_config: dict, calendar_tools: CalendarTools):
        self.base_url = llm_config.get("base_url", "http://localhost:11434")
        self.model = llm_config.get("model", "personaplex")
        self.temperature = llm_config.get("temperature", 0.7)
        self.max_tokens = llm_config.get("max_tokens", 512)

        self.assistant_name = assistant_config.get("name", "Ana")
        self.language = assistant_config.get("language", "es-MX")

        self.calendar_tools = calendar_tools
        self.system_prompt = get_full_system_prompt()

        # HTTP client for Ollama API
        self.client = httpx.AsyncClient(timeout=60.0)

    async def process(self, user_input: str, conversation_history: list) -> str:
        """
        Process user input and generate response

        Args:
            user_input: User's transcribed speech
            conversation_history: List of previous messages

        Returns:
            Assistant's response text
        """
        # Build messages for the API
        messages = self._build_messages(user_input, conversation_history)

        # Get response from LLM
        response = await self._call_ollama(messages)

        # Check for tool calls in response
        tool_result = await self._handle_tool_calls(response)

        if tool_result:
            # Add tool result to context and get final response
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "system", "content": f"Resultado de la herramienta: {tool_result}"})
            response = await self._call_ollama(messages)

        # Clean up response
        response = self._clean_response(response)

        return response

    def _build_messages(self, user_input: str, conversation_history: list) -> list:
        """Build message list for Ollama API"""
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        # Add conversation history (last 10 messages for context)
        for msg in conversation_history[-10:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        # Add current user input
        messages.append({"role": "user", "content": user_input})

        return messages

    async def _call_ollama(self, messages: list) -> str:
        """Call Ollama API for chat completion"""
        try:
            response = await self.client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    }
                }
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("message", {}).get("content", "")
            else:
                print(f"Ollama error: {response.status_code} - {response.text}")
                return "Disculpa, tuve un problema técnico. ¿Podrías repetir lo que dijiste?"

        except httpx.TimeoutException:
            return "Disculpa, me tardé mucho en responder. ¿Podrías repetir tu pregunta?"
        except Exception as e:
            print(f"LLM error: {e}")
            return "Disculpa, hubo un error. ¿Me puedes repetir lo que necesitas?"

    async def _handle_tool_calls(self, response: str) -> Optional[str]:
        """Extract and execute tool calls from response"""
        # Look for tool call pattern
        tool_pattern = r'<tool_call>\s*({.*?})\s*</tool_call>'
        match = re.search(tool_pattern, response, re.DOTALL)

        if not match:
            return None

        try:
            tool_data = json.loads(match.group(1))
            function_name = tool_data.get("function")
            arguments = tool_data.get("arguments", {})

            # Execute the tool
            result = await self.calendar_tools.execute(function_name, arguments)
            return result

        except json.JSONDecodeError as e:
            print(f"Tool call parse error: {e}")
            return "Error al procesar la solicitud"
        except Exception as e:
            print(f"Tool execution error: {e}")
            return f"Error: {str(e)}"

    def _clean_response(self, response: str) -> str:
        """Clean up response text for speech"""
        # Remove tool calls from response
        response = re.sub(r'<tool_call>.*?</tool_call>', '', response, flags=re.DOTALL)

        # Remove any remaining XML-like tags
        response = re.sub(r'<[^>]+>', '', response)

        # Remove markdown formatting
        response = re.sub(r'\*\*([^*]+)\*\*', r'\1', response)  # bold
        response = re.sub(r'\*([^*]+)\*', r'\1', response)      # italic
        response = re.sub(r'`([^`]+)`', r'\1', response)        # code

        # Clean up extra whitespace
        response = ' '.join(response.split())

        return response.strip()

    async def check_ollama_connection(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                return self.model in models or any(self.model in m for m in models)
            return False
        except:
            return False

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()
