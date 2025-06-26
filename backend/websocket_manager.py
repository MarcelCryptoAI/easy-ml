from fastapi import WebSocket
from typing import List
import json
import asyncio

class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except:
            self.disconnect(websocket)

    async def broadcast(self, message: dict):
        message_str = json.dumps(message)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except:
                disconnected.append(connection)
        
        for connection in disconnected:
            self.disconnect(connection)
    
    async def broadcast_trade_update(self, trade_data: dict):
        await self.broadcast({
            "type": "trade_update",
            "data": trade_data
        })
    
    async def broadcast_prediction_update(self, prediction_data: dict):
        await self.broadcast({
            "type": "prediction_update",
            "data": prediction_data
        })
    
    async def broadcast_position_update(self, position_data: dict):
        await self.broadcast({
            "type": "position_update",
            "data": position_data
        })