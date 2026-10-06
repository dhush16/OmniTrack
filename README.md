# OmniTrack: A Real-Time Industrial SOP Compliance Verification Platform

Real-time industrial Standard Operating Procedure (SOP) compliance verification using Protocol Directed Acyclic Graphs (DAG), acoustic intent grounding (Faster-Whisper), and visual verification.

## System Prerequisites
- Windows 10/11
- Python 3.10+
- NVIDIA GPU (RTX 40-series recommended) or CPU fallback
- Graphviz system binary installed and added to PATH

## Quick Setup

1. **Clone the repository:**
   \`\`\`bash
   git clone https://github.com/<YOUR_GITHUB_USERNAME>/OmniTrack.git
   cd OmniTrack
   \`\`\`

2. **Create and activate a virtual environment:**
   \`\`\`powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   \`\`\`

3. **Install dependencies:**
   \`\`\`powershell
   pip install -r requirements.txt
   \`\`\`

4. **Generate synthetic benchmark datasets:**
   \`\`\`powershell
   python generate_synthetic_trials.py
   \`\`\`

5. **Launch the platform:**
   - **Terminal 1 (Backend Telemetry & Video Engine):**
     \`\`\`powershell
     python server.py
     \`\`\`
   - **Terminal 2 (Supervisor Dashboard UI):**
     \`\`\`powershell
     streamlit run dashboard.py --server.port 8502
     \`\`\`
   - Open \`http://localhost:8502\` in your web browser.