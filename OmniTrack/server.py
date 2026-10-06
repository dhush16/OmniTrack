import json
import asyncio
import threading
import pyttsx3
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Initialize FastAPI App
app = FastAPI(title="OmniTrack Engine Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Offline TTS Voice Alert Engine
tts = pyttsx3.init()
tts.setProperty("rate", 180)

def speak_alert(msg: str):
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 180)
        engine.say(msg)
        engine.runAndWait()
    except Exception as e:
        print(f"TTS Error: {e}")

# Protocol DAG State Machine
class ProtocolDAG:
    def __init__(self, config_path="sop_assembly.json"):
        with open(config_path, "r") as f:
            self.schema = json.load(f)
        self.nodes = self.schema["nodes"]
        self.edges = set(self.schema["edges"])
        self.current_state = self.schema["initial_node"]
        self.visited = {self.current_state}
        self.ancestor_cache = self._precompute_ancestors()

    def _precompute_ancestors(self):
        ancestors = {n: set() for n in self.nodes}
        for node in self.nodes:
            queue = list(self.nodes[node].get("prerequisites", []))
            while queue:
                p = queue.pop(0)
                if p not in ancestors[node]:
                    ancestors[node].add(p)
                    queue.extend(self.nodes[p].get("prerequisites", []))
        return ancestors

    def evaluate(self, candidate_step: str):
        if candidate_step == self.current_state:
            return True, self.current_state, "IN_PROGRESS", []

        edge = f"{self.current_state}->{candidate_step}"
        if edge in self.edges:
            self.current_state = candidate_step
            self.visited.add(candidate_step)
            return True, self.current_state, "VALID_TRANSITION", []
        else:
            missing = list(self.ancestor_cache.get(candidate_step, set()) - self.visited)
            missing_names = [self.nodes[m]["title"] for m in missing if m in self.nodes]
            return False, self.current_state, "VIOLATION", missing_names

dag = ProtocolDAG()

# WebSocket Client Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active_connections:
            self.active_connections.remove(ws)

    async def broadcast(self, data: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(data)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    # Send current state upon connect
    await ws.send_json({
        "current_state": dag.current_state,
        "title": dag.nodes[dag.current_state]["title"],
        "status": "INITIALIZED",
        "missing": []
    })
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)

@app.post("/trigger_step/{step_id}")
async def trigger_step(step_id: str):
    valid, state, status, missing = dag.evaluate(step_id)
    payload = {
        "current_state": state,
        "title": dag.nodes[state]["title"],
        "status": status,
        "missing": missing
    }

    
    # Broadcast to Streamlit dashboard
    await manager.broadcast(payload)
    
    # Trigger Voice Preemption if Violation
    if not valid:
        alert_msg = f"Warning: {missing[0]} was skipped!" if missing else "Invalid step sequence!"
        threading.Thread(target=speak_alert, args=(alert_msg,), daemon=True).start()
        
    return payload

@app.get("/")
def read_root():
    return {
        "status": "online",
        "system": "OmniTrack Neuro-Symbolic Verification Engine",
        "active_state": dag.current_state,
        "docs_url": "http://127.0.0.1:8080/docs"
    }

if __name__ == "__main__":
    print("OmniTrack Telemetry Server starting on port 8000...")
    uvicorn.run(app, host="127.0.0.1", port=8080)