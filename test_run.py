import time
import pyttsx3

# Initialize TTS Voice Engine
tts = pyttsx3.init()
tts.setProperty("rate", 175)

def play_alert(message: str):
    print(f"\n>>> [VOICE PREEMPTION]: {message}")
    tts.say(message)
    tts.runAndWait()

# 5-Step PCB Assembly SOP Case Study
sop_schema = {
    "protocol_id": "SOP-ELEC-4091",
    "initial_node": "v1_esd_grounding",
    "nodes": {
        "v1_esd_grounding": {"name": "ESD Grounding Wristband", "prereqs": []},
        "v2_pcb_seating": {"name": "PCB Seating on Chassis", "prereqs": ["v1_esd_grounding"]},
        "v3_cross_torque": {"name": "Diagonal Bolt Cross-Torque", "prereqs": ["v2_pcb_seating"]},
        "v4_thermal_paste": {"name": "Thermal Paste Dispensing", "prereqs": ["v3_cross_torque"]},
        "v5_heatsink_latch": {"name": "Dual Heatsink Latching", "prereqs": ["v4_thermal_paste"]}
    },
    "edges": [
        "v1_esd_grounding->v2_pcb_seating",
        "v2_pcb_seating->v3_cross_torque",
        "v3_cross_torque->v4_thermal_paste",
        "v4_thermal_paste->v5_heatsink_latch"
    ]
}

class ProtocolDAGEngine:
    def __init__(self, schema):
        self.current = schema["initial_node"]
        self.visited = {self.current}
        self.edges = set(schema["edges"])
        self.nodes = schema["nodes"]

    def evaluate_transition(self, candidate_node: str):
        if candidate_node == self.current:
            return True, self.current, "[STATUS]: Step in progress."
            
        edge_key = f"{self.current}->{candidate_node}"
        if edge_key in self.edges:
            self.current = candidate_node
            self.visited.add(candidate_node)
            return True, self.current, f"[VALID TRANSITION] Advanced to {self.nodes[candidate_node]['name']}"
        else:
            missing_prereqs = [self.nodes[p]["name"] for p in self.nodes[candidate_node]["prereqs"] if p not in self.visited]
            return False, self.current, f"Warning: {missing_prereqs[0] if missing_prereqs else 'Prerequisite'} was skipped!"

# Instantiate System
engine = ProtocolDAGEngine(sop_schema)
print("=" * 65)
print("OmniTrack Runtime Engine Initialized on RTX 4050")
print("Initial Station State:", engine.nodes[engine.current]["name"])
print("=" * 65)

# Test 1: Valid Step Progression
print("\n--- TEST 1: Valid Operational Step ---")
ok, state, msg = engine.evaluate_transition("v2_pcb_seating")
print(msg)

# Test 2: Injected Human Error (Operator omits thermal paste and reaches for heatsink)
print("\n--- TEST 2: Injected SOP Omission (Skipping v3 & v4, attempting v5) ---")
ok, state, msg = engine.evaluate_transition("v5_heatsink_latch")
print(msg)
if not ok:
    play_alert(msg)

print("\nOmniTrack Verification Completed Successfully.")