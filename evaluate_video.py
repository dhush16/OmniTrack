import cv2
import json
import time
import os

with open("sop_assembly.json", "r") as f:
    sop = json.load(f)

nodes = sop["nodes"]
edges = set(sop["edges"])
initial = sop["initial_node"]

ancestors = {n: set() for n in nodes}
for node in nodes:
    q = list(nodes[node].get("prerequisites", []))
    while q:
        p = q.pop(0)
        if p not in ancestors[node]:
            ancestors[node].add(p)
            q.extend(nodes[p].get("prerequisites", []))

def run_evaluation(video_path, mock_detected_sequence):
    print(f"\n==================================================")
    print(f"EVALUATING: {video_path}")
    print(f"==================================================")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = int(fps * 0.5)  # Delta_t = 0.5s sampling (2 FPS)
    
    current_state = initial
    visited = {current_state}
    frame_count = 0
    step_idx = 0
    
    violations_detected = 0
    total_steps = len(mock_detected_sequence)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        # Process keyframe every 0.5 seconds
        if frame_count % frame_interval == 0 and step_idx < total_steps:
            proposed_step = mock_detected_sequence[step_idx]
            step_idx += 1
            
            edge = f"{current_state}->{proposed_step}"
            if proposed_step == current_state:
                status = "IN_PROGRESS"
            elif edge in edges:
                current_state = proposed_step
                visited.add(proposed_step)
                status = f"VALID -> {proposed_step}"
            else:
                missing = list(ancestors.get(proposed_step, set()) - visited)
                violations_detected += 1
                status = f"VIOLATION! Skipped: {missing}"
            
            timestamp_sec = frame_count / fps
            print(f"[{timestamp_sec:4.1f}s] Proposed: {proposed_step:18} | Result: {status}")

    cap.release()
    print(f"Evaluation complete. Total Steps Evaluated: {total_steps}, Violations Intercepted: {violations_detected}")

# Test 1: Normal execution
seq_normal = ["v1_esd_grounding", "v2_pcb_seating", "v3_cross_torque", "v4_thermal_paste", "v5_heatsink_latch"]
run_evaluation("data/trial_1_normal.mp4", seq_normal)

# Test 2: Injected Omission (Skipping v4 paste)
seq_omission = ["v1_esd_grounding", "v2_pcb_seating", "v3_cross_torque", "v5_heatsink_latch"]
run_evaluation("data/trial_2_omission.mp4", seq_omission)