from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from trickster import Trickster

# App section

DESCRIPTION = """
    Trickster chatbot backend implemented with FastAPI websockets.
"""

app: FastAPI = FastAPI(
    title="Trickster Chatbot Backend",
    description=DESCRIPTION,
    version="0.1.0",
    contact={
        "name": "mal2",
        "url": "https://github.com/mal2/python-chatbot-api",
    },
)

origins = [
    "http://localhost",
    "http://127.0.0.1:5000",
    "http://cbapi.up.railway.app",
    "https://cbapi.up.railway.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class ConnectionManager:
    """
    Connection manager for websockets.
    Get from FastAPI documentation.
    """

    def __init__(self) -> None:
        """
        Constructor.
        """
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """
        Create websocket connection and add to
        active connection list.
        Args:
          websocket: A Websocket instance to add.
        """
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        """
        Remove websocket connection from active
        connections list.
        Args:
          websocket: A Websocket instance to remove.
        """
        self.active_connections.remove(websocket)

    async def reply(self, message: str, websocket: WebSocket) -> None:
        """
        Send text message to websocket connection.
        Args:
          message: Text message to send.
          websocket: A Websocket instance addressee.
        """
        if message:
            await websocket.send_text(message)

    async def quit(self, message: str, websocket: WebSocket) -> None:
        """
        Send farewell message and disconnect.
        Args:
          message: Farewell text message to send.
          websocket: A Websocket instance addressee and to disconnect.
        """
        await self.reply(message, websocket)
        await websocket.close()
        self.disconnect(websocket)


manager: ConnectionManager = ConnectionManager()
trickster: Trickster = Trickster("./data.pth", "./intents.json")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    #if websocket.headers["origin"] == environ["ALLOWED_ORIGIN"]:
    await manager.connect(websocket)
    try:
        await manager.reply(trickster.greeting(), websocket)
        while True:
            data: str = await websocket.receive_text()
            if data == "quit":
                await manager.quit(trickster.response(data), websocket)
                break
            else:
                await manager.reply(trickster.response(data), websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)