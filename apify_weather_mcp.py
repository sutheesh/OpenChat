"""
Apify Weather MCP Client - Direct Actor Endpoint
Uses: https://jiri-spilka--weather-mcp-server.apify.actor/mcp
"""

import httpx
import json
from typing import Optional, Tuple


class ApifyWeatherMCP:
    """
    Connects directly to the weather Actor's MCP endpoint
    """
    
    def __init__(self, apify_token: Optional[str] = None):
        """Initialize Apify Weather MCP client"""
        if not apify_token:
            print("⚠️  Warning: No APIFY_TOKEN provided.")
            print("   Get free token at: https://console.apify.com/")
        
        self.apify_token = apify_token
        
        # Direct Actor endpoint (not the general Apify platform)
        self.base_url = "https://jiri-spilka--weather-mcp-server.apify.actor/mcp"
        self.session_id = None
        
        print(f"✓ Apify Weather MCP initialized")
        print(f"  Endpoint: jiri-spilka--weather-mcp-server.apify.actor")
    
    def _get_headers(self, include_session=False):
        """Build request headers"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        
        if self.apify_token:
            headers["Authorization"] = f"Bearer {self.apify_token}"
        
        if include_session and self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        
        return headers
    
    def _parse_sse_response(self, text: str) -> Optional[dict]:
        """Parse Server-Sent Events format"""
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('data: '):
                json_str = line[6:]
                
                try:
                    data = json.loads(json_str)
                    if "result" in data or "error" in data:
                        return data
                except json.JSONDecodeError:
                    continue
        
        return None
    
    async def _initialize_session(self, client: httpx.AsyncClient):
        """Initialize MCP session"""
        if self.session_id:
            return
        
        init_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {
                    "name": "qwen-weather-client",
                    "version": "1.0.0"
                }
            }
        }
        
        print(f"  → Initializing session...")
        
        response = await client.post(
            self.base_url,
            json=init_payload,
            headers=self._get_headers(include_session=False)
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Init failed: {response.status_code} - {response.text}")
        
        # Parse response
        content_type = response.headers.get("Content-Type", "")
        
        if "text/event-stream" in content_type:
            data = self._parse_sse_response(response.text)
        else:
            data = response.json()
        
        if not data:
            raise RuntimeError("Could not parse initialization response")
        
        # Extract session ID from header
        session_id = response.headers.get("Mcp-Session-Id")
        
        if not session_id:
            raise RuntimeError("Server did not return session ID")
        
        self.session_id = session_id
        print(f"  ✓ Session ID: {self.session_id[:16]}...")
        
        # Send initialized notification
        await client.post(
            self.base_url,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers=self._get_headers(include_session=True)
        )
        
        print(f"  ✓ Session initialized")
    
    async def get_weather(self, city: str) -> str:
        """Get current weather for a city"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Initialize session
                await self._initialize_session(client)
                
                # Call weather tool
                # Tool name is "get_current_weather" (from the Actor)
                payload = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {
                        "name": "get_current_weather",  # Correct tool name
                        "arguments": {
                            "city": city  # Parameter is "city" not "location"
                        }
                    }
                }
                
                print(f"  → Requesting weather for: {city}")
                
                response = await client.post(
                    self.base_url,
                    json=payload,
                    headers=self._get_headers(include_session=True)
                )
                
                print(f"  ← Status: {response.status_code}")
                
                if response.status_code != 200:
                    return f"Error: HTTP {response.status_code} - {response.text[:200]}"
                
                # Parse response
                content_type = response.headers.get("Content-Type", "")
                
                if "text/event-stream" in content_type:
                    data = self._parse_sse_response(response.text)
                else:
                    try:
                        data = response.json()
                    except json.JSONDecodeError:
                        return f"Error: Invalid JSON - {response.text[:200]}"
                
                if not data:
                    return f"Error: Could not parse response - {response.text[:200]}"
                
                # Check for error
                if "error" in data:
                    error_msg = data["error"].get("message", str(data["error"]))
                    
                    # If tool not found, try listing tools
                    if "not found" in error_msg.lower():
                        # List tools to see what's available
                        tools_payload = {
                            "jsonrpc": "2.0",
                            "id": "list-tools",
                            "method": "tools/list",
                            "params": {}
                        }
                        
                        tools_response = await client.post(
                            self.base_url,
                            json=tools_payload,
                            headers=self._get_headers(include_session=True)
                        )
                        
                        if tools_response.status_code == 200:
                            content_type = tools_response.headers.get("Content-Type", "")
                            
                            if "text/event-stream" in content_type:
                                tools_data = self._parse_sse_response(tools_response.text)
                            else:
                                tools_data = tools_response.json()
                            
                            if tools_data and "result" in tools_data:
                                result = tools_data["result"]
                                if isinstance(result, dict) and "tools" in result:
                                    tool_names = [t.get("name", "") for t in result["tools"]]
                                    return f"Tool 'get_weather' not found. Available tools: {', '.join(tool_names)}"
                    
                    if "session" in error_msg.lower():
                        self.session_id = None
                        return await self.get_weather(city)
                    
                    return f"Weather error: {error_msg}"
                
                # Extract result
                if "result" in data:
                    result = data["result"]
                    
                    # Handle content array format
                    if isinstance(result, dict) and "content" in result:
                        content = result["content"]
                        
                        if isinstance(content, list):
                            text_parts = []
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    text_parts.append(item.get("text", ""))
                            
                            if text_parts:
                                print(f"  ✓ Got weather data")
                                return "\n".join(text_parts)
                    
                    # Direct string result
                    elif isinstance(result, str):
                        print(f"  ✓ Got weather data")
                        return result
                
                # Debug: show what we got
                return f"Unexpected format: {json.dumps(data, indent=2)[:500]}"
        
        except httpx.TimeoutException:
            return f"Request timed out for {city}"
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            
            if "session" in str(e).lower():
                self.session_id = None
            
            return f"Error: {str(e)}"
    
    def detect_weather_query(self, message: str) -> Optional[str]:
        """Detect if message is asking about weather"""
        message_lower = message.lower()
        
        weather_keywords = [
            'weather', 'temperature', 'forecast', 'rain', 'sunny', 
            'cloudy', 'climate', 'hot', 'cold', 'warm', 'cool'
        ]
        
        if not any(keyword in message_lower for keyword in weather_keywords):
            return None
        
        words = message.split()
        
        # Look for "in [city]"
        for i, word in enumerate(words):
            if word.lower() in ['in', 'at', 'for'] and i + 1 < len(words):
                city = words[i + 1].strip('.,!?')
                return city.title()
        
        # Common cities
        common_cities = {
            'london': 'London', 'paris': 'Paris', 'tokyo': 'Tokyo',
            'newyork': 'New York', 'new york': 'New York',
            'sydney': 'Sydney', 'berlin': 'Berlin', 'moscow': 'Moscow',
            'beijing': 'Beijing', 'mumbai': 'Mumbai', 'delhi': 'Delhi',
            'losangeles': 'Los Angeles', 'los angeles': 'Los Angeles',
            'chicago': 'Chicago', 'toronto': 'Toronto', 'boston': 'Boston',
            'seattle': 'Seattle', 'miami': 'Miami', 'dallas': 'Dallas',
            'houston': 'Houston', 'atlanta': 'Atlanta', 'phoenix': 'Phoenix'
        }
        
        for city_key, city_name in common_cities.items():
            if city_key in message_lower.replace(' ', ''):
                return city_name
        
        return None
    
    async def process_message(self, message: str) -> Tuple[Optional[str], bool]:
        """Process message and get weather if needed"""
        city = self.detect_weather_query(message)
        
        if city:
            print(f"  🌤️  Detected weather query for: {city}")
            weather_data = await self.get_weather(city)
            return (weather_data, True)
        
        return (None, False)


# Global instance
_apify_weather = None

def get_apify_weather(apify_token: Optional[str] = None) -> ApifyWeatherMCP:
    """Get or create global Apify weather client"""
    global _apify_weather
    if _apify_weather is None:
        _apify_weather = ApifyWeatherMCP(apify_token)
    return _apify_weather


# Flask integration
async def process_weather_query(message: str, apify_token: Optional[str] = None) -> Tuple[Optional[str], bool]:
    """Process message for weather queries"""
    client = get_apify_weather(apify_token)
    return await client.process_message(message)