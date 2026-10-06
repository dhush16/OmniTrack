import os
import cv2
import numpy as np
import time
import math

# Output configuration
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TRIAL_1_PATH = os.path.join(OUTPUT_DIR, "trial_1_normal.mp4")
TRIAL_2_PATH = os.path.join(OUTPUT_DIR, "trial_2_omission.mp4")

WIDTH = 1280
HEIGHT = 720
FPS = 30.0

# SOP Node Definitions
NODES = {
    "v1_esd_grounding": "Step 1: ESD Wristband Grounding",
    "v2_pcb_seating": "Step 2: PCB Motherboard Seating",
    "v3_cross_torque": "Step 3: 4-Corner Diagonal Bolt Torque",
    "v4_thermal_paste": "Step 4: Thermal Paste Dispensing",
    "v5_heatsink_latch": "Step 5: Dual Heatsink Cooler Latching"
}

def draw_hud(frame, trial_name, step_id, step_title, elapsed_sec, total_sec, frame_idx, total_frames, is_violation=False):
    """Draws top HUD status banner and bottom telemetry readout."""
    # Top HUD background
    header_color = (20, 24, 32)
    border_color = (60, 68, 80) if not is_violation else (40, 40, 220)
    cv2.rectangle(frame, (0, 0), (WIDTH, 85), header_color, -1)
    cv2.line(frame, (0, 85), (WIDTH, 85), border_color, 2)

    # Title & Protocol
    cv2.putText(frame, "OMNITRACK | SYNTHETIC SOP WORKSTATION BENCHMARK", (24, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (232, 181, 41), 2)
    cv2.putText(frame, f"Protocol: SOP-ELEC-4091 | Scenario: {trial_name}", (24, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 190, 205), 1)

    # Timestamp & Frame Info (Right-aligned)
    time_str = f"Time: {elapsed_sec:04.1f}s / {total_sec:04.1f}s"
    frame_str = f"Frame: {frame_idx:03d} / {total_frames:03d} (30 FPS)"
    cv2.putText(frame, time_str, (WIDTH - 300, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 245, 250), 2)
    cv2.putText(frame, frame_str, (WIDTH - 300, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 155, 175), 1)

    # Bottom Status Bar
    footer_color = (18, 22, 28)
    cv2.rectangle(frame, (0, HEIGHT - 55), (WIDTH, HEIGHT), footer_color, -1)
    cv2.line(frame, (0, HEIGHT - 55), (WIDTH, HEIGHT - 55), border_color, 2)

    # Active Step Info
    step_label = f"ACTIVE NODE: [{step_id}] - {step_title.upper()}"
    cv2.putText(frame, step_label, (24, HEIGHT - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 245, 255), 2)

    # Compliance Status Badge
    if is_violation:
        badge_bg = (30, 30, 210)
        badge_text = "STATUS: SOP OMISSION VIOLATION DETECTED"
        cv2.rectangle(frame, (WIDTH - 440, HEIGHT - 46), (WIDTH - 20, HEIGHT - 10), badge_bg, -1)
        cv2.putText(frame, badge_text, (WIDTH - 425, HEIGHT - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
    else:
        badge_bg = (34, 139, 34)
        badge_text = "STATUS: VERIFIED SEQUENCE (COMPLIANT)"
        cv2.rectangle(frame, (WIDTH - 420, HEIGHT - 46), (WIDTH - 20, HEIGHT - 10), badge_bg, -1)
        cv2.putText(frame, badge_text, (WIDTH - 405, HEIGHT - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

def draw_workstation_mat(frame):
    """Draws workstation anti-static surface and inspection bounding zone."""
    # Matte surface
    frame[:] = (32, 38, 46)

    # Grid lines on ESD mat
    for x in range(100, WIDTH, 80):
        cv2.line(frame, (x, 90), (x, HEIGHT - 60), (38, 45, 54), 1)
    for y in range(90, HEIGHT - 60, 60):
        cv2.line(frame, (100, y), (WIDTH - 100, y), (38, 45, 54), 1)

    # Workstation Assembly Fixture Jig
    jig_x1, jig_y1, jig_x2, jig_y2 = 240, 110, 1040, 640
    cv2.rectangle(frame, (jig_x1, jig_y1), (jig_x2, jig_y2), (48, 56, 68), -1)
    cv2.rectangle(frame, (jig_x1, jig_y1), (jig_x2, jig_y2), (72, 85, 102), 2)

    # Fixture corner alignment brackets
    corner_size = 28
    # Top-Left
    cv2.rectangle(frame, (jig_x1, jig_y1), (jig_x1 + corner_size, jig_y1 + corner_size), (90, 105, 125), -1)
    # Top-Right
    cv2.rectangle(frame, (jig_x2 - corner_size, jig_y1), (jig_x2, jig_y1 + corner_size), (90, 105, 125), -1)
    # Bottom-Left
    cv2.rectangle(frame, (jig_x1, jig_y2 - corner_size), (jig_x1 + corner_size, jig_y2), (90, 105, 125), -1)
    # Bottom-Right
    cv2.rectangle(frame, (jig_x2 - corner_size, jig_y2 - corner_size), (jig_x2, jig_y2), (90, 105, 125), -1)

    # Optical Inspection Camera Overlay Box
    cv2.rectangle(frame, (280, 130), (1000, 620), (0, 220, 120), 2)
    cv2.putText(frame, "WORKSTATION INSPECTION ZONE (B_v)", (290, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 120), 2)

def draw_esd_lead(frame, t):
    """Draws ESD grounding terminal, coiled lead wire, and wristband."""
    term_x, term_y = 170, 200
    # Ground terminal lug (Brass / Gold)
    cv2.circle(frame, (term_x, term_y), 18, (30, 180, 240), -1)
    cv2.circle(frame, (term_x, term_y), 8, (20, 24, 30), -1)
    # Yellow ESD Warning Triangle & Ground Symbol
    cv2.putText(frame, "GND", (term_x - 16, term_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 2)

    # Coiled Grounding Lead (Spring effect to wristband)
    wrist_x, wrist_y = 170, 480
    prev_pt = (term_x, term_y + 18)
    coils = 14
    for i in range(1, coils + 1):
        progress = i / coils
        curr_y = int(term_y + 18 + progress * (wrist_y - term_y - 45))
        offset = 12 if i % 2 == 1 else -12
        curr_pt = (term_x + offset, curr_y)
        cv2.line(frame, prev_pt, curr_pt, (230, 140, 20), 3)
        prev_pt = curr_pt
    cv2.line(frame, prev_pt, (wrist_x, wrist_y - 25), (230, 140, 20), 3)

    # Blue Operator Wristband
    cv2.rectangle(frame, (wrist_x - 30, wrist_y - 25), (wrist_x + 30, wrist_y + 25), (210, 100, 25), -1)
    cv2.rectangle(frame, (wrist_x - 30, wrist_y - 25), (wrist_x + 30, wrist_y + 25), (250, 200, 80), 2)
    # Wristband snap button
    cv2.circle(frame, (wrist_x, wrist_y), 7, (200, 200, 200), -1)

    # Grounding status indicator (blinking green pulse)
    pulse = int(128 + 127 * math.sin(t * 6.0))
    cv2.circle(frame, (term_x, term_y), 24, (0, pulse, 0), 2)
    cv2.putText(frame, "ESD GROUNDED: 0.98 M-Ohm", (80, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1)

def draw_pcb_motherboard(frame):
    """Draws the seated PCB Motherboard with socket, traces, and solder pads."""
    pcb_x1, pcb_y1, pcb_x2, pcb_y2 = 420, 200, 860, 560
    # Emerald green PCB substrate
    cv2.rectangle(frame, (pcb_x1, pcb_y1), (pcb_x2, pcb_y2), (28, 88, 48), -1)
    cv2.rectangle(frame, (pcb_x1, pcb_y1), (pcb_x2, pcb_y2), (40, 130, 70), 3)

    # Silkscreen text and markings
    cv2.putText(frame, "OMNITRACK PCB REV 4.2 | LGA-1700 SOCKET", (pcb_x1 + 20, pcb_y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 235, 220), 1)

    # RAM Slots (DDR5) on left side of CPU
    for rx in [pcb_x1 + 30, pcb_x1 + 50, pcb_x1 + 70]:
        cv2.rectangle(frame, (rx, pcb_y1 + 70), (rx + 12, pcb_y2 - 70), (18, 22, 28), -1)
        cv2.rectangle(frame, (rx, pcb_y1 + 70), (rx + 12, pcb_y2 - 70), (180, 190, 80), 1)

    # PCIe Slot below CPU
    cv2.rectangle(frame, (pcb_x1 + 140, pcb_y2 - 50), (pcb_x2 - 40, pcb_y2 - 32), (18, 22, 28), -1)
    cv2.rectangle(frame, (pcb_x1 + 140, pcb_y2 - 50), (pcb_x2 - 40, pcb_y2 - 32), (200, 200, 200), 1)

    # CPU Socket Housing (Center)
    sock_x1, sock_y1, sock_x2, sock_y2 = 580, 310, 720, 450
    cv2.rectangle(frame, (sock_x1, sock_y1), (sock_x2, sock_y2), (180, 185, 195), -1)
    cv2.rectangle(frame, (sock_x1, sock_y1), (sock_x2, sock_y2), (100, 105, 115), 2)

    # Gold Pin Grid inside CPU socket
    cv2.rectangle(frame, (sock_x1 + 15, sock_y1 + 15), (sock_x2 - 15, sock_y2 - 15), (40, 140, 180), -1)

    # CPU Silicon Integrated Heat Spreader (IHS)
    cv2.rectangle(frame, (605, 335), (695, 425), (140, 145, 155), -1)
    cv2.rectangle(frame, (605, 335), (695, 425), (200, 205, 215), 2)
    cv2.putText(frame, "CPU", (632, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 235, 240), 2)

    # 4 Corner Bolt Mounting Holes
    holes = [(450, 230), (830, 230), (450, 530), (830, 530)]
    for hx, hy in holes:
        cv2.circle(frame, (hx, hy), 12, (20, 24, 28), -1)
        cv2.circle(frame, (hx, hy), 12, (180, 185, 120), 2)

def draw_bolts_and_torque(frame, t):
    """Draws 4 corner bolts and animated screwdriver tool torquing diagonal bolts."""
    holes = [
        ("TL", (450, 230)),
        ("BR", (830, 530)),
        ("TR", (830, 230)),
        ("BL", (450, 530)),
    ]

    # Torquing sequence (Diagonal 1: TL -> BR, Diagonal 2: TR -> BL)
    # t ranges from 0.0 to 5.0
    active_idx = min(3, int(t / 1.25))

    for idx, (label, (hx, hy)) in enumerate(holes):
        is_done = idx < active_idx or t >= 4.8
        is_current = idx == active_idx and t < 4.8

        if is_done:
            # Torqued Bolt with green ring
            cv2.circle(frame, (hx, hy), 9, (120, 125, 135), -1)
            cv2.line(frame, (hx - 5, hy), (hx + 5, hy), (50, 55, 60), 2)
            cv2.line(frame, (hx, hy - 5), (hx, hy + 5), (50, 55, 60), 2)
            cv2.circle(frame, (hx, hy), 15, (0, 220, 80), 2)
            cv2.putText(frame, f"0.6Nm", (hx - 22, hy - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 80), 1)
        elif is_current:
            # Active Torquing with spinning angle
            cv2.circle(frame, (hx, hy), 9, (160, 165, 175), -1)
            spin_angle = (t * 15.0) % (2 * math.pi)
            dx = int(6 * math.cos(spin_angle))
            dy = int(6 * math.sin(spin_angle))
            cv2.line(frame, (hx - dx, hy - dy), (hx + dx, hy + dy), (40, 45, 50), 2)
            cv2.circle(frame, (hx, hy), 18, (0, 180, 255), 2)

            # Screwdriver representation above active bolt
            driver_x, driver_y = hx + 40, hy - 50
            # Driver shaft
            cv2.line(frame, (hx, hy), (driver_x, driver_y), (160, 170, 180), 6)
            # Driver handle
            cv2.line(frame, (driver_x, driver_y), (driver_x + 35, driver_y - 45), (30, 80, 220), 14)
            cv2.putText(frame, "TORQUING...", (hx + 15, hy + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 2)
        else:
            # Untorqued bolt placed
            cv2.circle(frame, (hx, hy), 9, (100, 105, 115), -1)
            cv2.line(frame, (hx - 4, hy), (hx + 4, hy), (50, 55, 60), 2)
            cv2.line(frame, (hx, hy - 4), (hx, hy + 4), (50, 55, 60), 2)

def draw_thermal_paste(frame, t):
    """Draws syringe dispenser tool and dispensing of cross-pattern thermal compound."""
    # Always keep 4 bolts secured
    for hx, hy in [(450, 230), (830, 530), (830, 230), (450, 530)]:
        cv2.circle(frame, (hx, hy), 9, (120, 125, 135), -1)
        cv2.circle(frame, (hx, hy), 15, (0, 220, 80), 2)

    cx, cy = 650, 380  # Center of CPU die
    dispense_progress = min(1.0, t / 4.0)

    # Draw cross-pattern ("X") thermal compound
    arm_length = int(32 * dispense_progress)
    if arm_length > 2:
        paste_color = (200, 205, 215)
        paste_shadow = (140, 145, 155)
        # Line 1: Top-Left to Bottom-Right
        cv2.line(frame, (cx - arm_length, cy - arm_length), (cx + arm_length, cy + arm_length), paste_shadow, 8)
        cv2.line(frame, (cx - arm_length, cy - arm_length), (cx + arm_length, cy + arm_length), paste_color, 5)
        # Line 2: Bottom-Left to Top-Right
        cv2.line(frame, (cx - arm_length, cy + arm_length), (cx + arm_length, cy - arm_length), paste_shadow, 8)
        cv2.line(frame, (cx - arm_length, cy + arm_length), (cx + arm_length, cy - arm_length), paste_color, 5)
        # Center bead
        cv2.circle(frame, (cx, cy), 9, paste_color, -1)

    # Thermal Paste Syringe Dispenser tool
    if t < 4.5:
        syr_tip_x = cx + int(15 * math.cos(t * 4))
        syr_tip_y = cy + int(10 * math.sin(t * 4)) - 10
        syr_body_x = syr_tip_x + 60
        syr_body_y = syr_tip_y - 75

        # Needle nozzle
        cv2.line(frame, (syr_tip_x, syr_tip_y), (syr_tip_x + 18, syr_tip_y - 22), (180, 185, 190), 3)
        # Syringe body
        cv2.line(frame, (syr_tip_x + 18, syr_tip_y - 22), (syr_body_x, syr_body_y), (230, 235, 240), 12)
        cv2.line(frame, (syr_tip_x + 18, syr_tip_y - 22), (syr_body_x, syr_body_y), (30, 120, 200), 8)
        # Plunger
        cv2.line(frame, (syr_body_x, syr_body_y), (syr_body_x + 25, syr_body_y - 30), (140, 145, 150), 6)
        cv2.putText(frame, "DISPENSING THERMAL PASTE", (cx - 110, cy - 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 2)

    cv2.putText(frame, "TIM VOLUME: 0.25 mL (CROSS PATTERN)", (cx - 120, cy + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 230, 240), 1)

def draw_heatsink_cooler(frame, t, paste_applied=True):
    """Draws cooler seated over CPU die with dual latch bars."""
    # Keep bolts secured
    for hx, hy in [(450, 230), (830, 530), (830, 230), (450, 530)]:
        cv2.circle(frame, (hx, hy), 9, (120, 125, 135), -1)
        cv2.circle(frame, (hx, hy), 15, (0, 220, 80), 2)

    # Cooler bounding box
    hx1, hy1, hx2, hy2 = 560, 290, 740, 470

    # If omission (no paste applied), show warning banner and dry CPU die
    if not paste_applied:
        # Alert box over CPU socket
        cv2.rectangle(frame, (530, 270), (770, 490), (30, 30, 180), 2)
        cv2.putText(frame, "! TIM OMITTED !", (580, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 255), 2)

    # Aluminum heatsink fin block
    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (170, 175, 185), -1)
    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (100, 105, 115), 3)

    # Horizontal cooling fin lines
    for fy in range(hy1 + 14, hy2, 12):
        cv2.line(frame, (hx1 + 10, fy), (hx2 - 10, fy), (90, 95, 105), 2)

    # Center copper heat pipe caps
    for cx in [620, 650, 680]:
        cv2.circle(frame, (cx, hy1 + 25), 7, (40, 120, 200), -1)
        cv2.circle(frame, (cx, hy2 - 25), 7, (40, 120, 200), -1)

    # Dual Latch Bars on Left and Right
    latch_progress = min(1.0, t / 3.0)
    # Left latch lever
    left_lever_y = int(hy1 + 40 + latch_progress * 50)
    cv2.rectangle(frame, (hx1 - 25, hy1 + 35), (hx1, hy2 - 35), (40, 45, 55), -1)
    cv2.line(frame, (hx1 - 25, left_lever_y), (hx1 - 50, left_lever_y + 15), (200, 205, 215), 5)
    # Right latch lever
    right_lever_y = int(hy1 + 40 + latch_progress * 50)
    cv2.rectangle(frame, (hx2, hy1 + 35), (hx2 + 25, hy2 - 35), (40, 45, 55), -1)
    cv2.line(frame, (hx2 + 25, right_lever_y), (hx2 + 50, right_lever_y + 15), (200, 205, 215), 5)

    if paste_applied:
        cv2.putText(frame, "DUAL COOLER LATCHES SECURED & CLAMPED", (520, hy2 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 120), 2)
    else:
        # Prominent Omission Violation Callout
        cv2.putText(frame, "CRITICAL ERROR: HEATSINK MOUNTED ON DRY CPU DIE", (470, hy2 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (50, 50, 255), 2)
        cv2.putText(frame, "SKIPPED PREREQUISITE: v4_thermal_paste", (510, hy2 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 2)

def generate_trial_video(output_file, steps_config, trial_title, is_omission=False):
    """Generates a synthetic assembly trial MP4 video."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, FPS, (WIDTH, HEIGHT))

    if not out.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {output_file}")

    total_sec = sum(dur for _, _, dur in steps_config)
    total_frames = int(total_sec * FPS)

    print(f"Generating '{output_file}' ({total_sec}s, {total_frames} frames @ {FPS} FPS)...")
    global_frame = 0

    for step_idx, (step_id, step_title, dur_sec) in enumerate(steps_config):
        step_frames = int(dur_sec * FPS)
        for sf in range(step_frames):
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            t_step = sf / FPS
            t_total = global_frame / FPS

            # 1. Base Workstation Mat & Inspection Fixture
            draw_workstation_mat(frame)

            # 2. Sequential Visual Layering
            # Step 1: ESD wristband lead (visible in all subsequent steps)
            draw_esd_lead(frame, t_total)

            # Step 2: PCB Motherboard
            if step_id in ["v2_pcb_seating", "v3_cross_torque", "v4_thermal_paste", "v5_heatsink_latch"]:
                draw_pcb_motherboard(frame)

            # Step 3: Bolt Torquing
            if step_id == "v3_cross_torque":
                draw_bolts_and_torque(frame, t_step)
            elif step_id in ["v4_thermal_paste", "v5_heatsink_latch"]:
                # Bolts already torqued
                for hx, hy in [(450, 230), (830, 530), (830, 230), (450, 530)]:
                    cv2.circle(frame, (hx, hy), 9, (120, 125, 135), -1)
                    cv2.circle(frame, (hx, hy), 15, (0, 220, 80), 2)

            # Step 4: Thermal Paste Dispensing
            if step_id == "v4_thermal_paste":
                draw_thermal_paste(frame, t_step)

            # Step 5: Heatsink Cooler Latching
            if step_id == "v5_heatsink_latch":
                draw_heatsink_cooler(frame, t_step, paste_applied=(not is_omission))

            # 3. Top HUD Banner and Bottom Readout
            is_step_violation = is_omission and (step_id == "v5_heatsink_latch")
            draw_hud(frame, trial_title, step_id, step_title, t_total, total_sec,
                     global_frame + 1, total_frames, is_violation=is_step_violation)

            out.write(frame)
            global_frame += 1

    out.release()
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"-> Successfully saved: {output_file} ({file_size_mb:.2f} MB)")

def main():
    print("==================================================")
    print("OmniTrack Synthetic SOP Trial Generator Active")
    print(f"Target Directory: {OUTPUT_DIR}")
    print("==================================================\n")

    # Trial 1: Normal 5-Step Sequential Assembly (~25s)
    steps_trial_1 = [
        ("v1_esd_grounding", NODES["v1_esd_grounding"], 5.0),
        ("v2_pcb_seating", NODES["v2_pcb_seating"], 5.0),
        ("v3_cross_torque", NODES["v3_cross_torque"], 5.0),
        ("v4_thermal_paste", NODES["v4_thermal_paste"], 5.0),
        ("v5_heatsink_latch", NODES["v5_heatsink_latch"], 5.0),
    ]
    generate_trial_video(TRIAL_1_PATH, steps_trial_1, "Trial 1 [Normal Execution]", is_omission=False)

    # Trial 2: Injected SOP Omission (~20s, skips Step 4)
    steps_trial_2 = [
        ("v1_esd_grounding", NODES["v1_esd_grounding"], 5.0),
        ("v2_pcb_seating", NODES["v2_pcb_seating"], 5.0),
        ("v3_cross_torque", NODES["v3_cross_torque"], 5.0),
        ("v5_heatsink_latch", NODES["v5_heatsink_latch"], 5.0),  # Skipped v4!
    ]
    generate_trial_video(TRIAL_2_PATH, steps_trial_2, "Trial 2 [Omission Anomaly - Skipped v4]", is_omission=True)

    print("\n==================================================")
    print("Synthetic Trial Generation Completed Successfully!")
    print(f"1. {TRIAL_1_PATH}")
    print(f"2. {TRIAL_2_PATH}")
    print("==================================================")

if __name__ == "__main__":
    main()
