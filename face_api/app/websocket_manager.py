import asyncio
import json
from typing import Dict, Set
from datetime import datetime


class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, Set] = {}

    async def connect(self, camera_id: str, websocket):
        if camera_id not in self.active_connections:
            self.active_connections[camera_id] = set()
        self.active_connections[camera_id].add(websocket)

    async def disconnect(self, camera_id: str, websocket):
        if camera_id in self.active_connections:
            self.active_connections[camera_id].discard(websocket)
            if not self.active_connections[camera_id]:
                del self.active_connections[camera_id]

    async def broadcast(self, camera_id: str, message: dict):
        if camera_id not in self.active_connections:
            return

        message["timestamp"] = datetime.now().isoformat()
        message_json = json.dumps(message)

        disconnected = set()
        for websocket in self.active_connections[camera_id]:
            try:
                await websocket.send_text(message_json)
            except Exception:
                disconnected.add(websocket)

        for ws in disconnected:
            await self.disconnect(camera_id, ws)

    def get_connected_clients(self, camera_id: str) -> int:
        return len(self.active_connections.get(camera_id, set()))


websocket_manager = WebSocketManager()