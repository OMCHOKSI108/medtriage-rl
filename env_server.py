from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from models import Reward, StateResponse, StepRequest, StepResponse
from src.medtriage.sim import MedTriageSim

app = FastAPI(title="MedTriage ER Simulator")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "details": str(exc)},
    )

_sim = MedTriageSim()


@app.post("/reset", response_model=StepResponse)
def reset() -> StepResponse:
    observation = _sim.reset()
    return StepResponse(
        observation=observation,
        reward=Reward(value=0.0, components={"reset": 0.0}),
        done=False,
        info={"note": "reset"},
    )


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest) -> StepResponse:
    observation, reward, done, info = _sim.step(request.action)
    return StepResponse(
        observation=observation,
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    return StateResponse(state=_sim.state())


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MedTriage Mission Control</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #0f172a;
                --card: #1e293b;
                --accent: #38bdf8;
                --success: #22c55e;
                --warning: #f59e0b;
                --danger: #ef4444;
                --text: #f8fafc;
                --text-dim: #94a3b8;
            }
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                background: var(--bg); 
                color: var(--text); 
                font-family: 'Inter', sans-serif; 
                line-height: 1.5;
                padding: 2rem;
            }
            .dashboard {
                max-width: 1200px;
                margin: 0 auto;
                display: grid;
                grid-template-columns: 2fr 1fr;
                gap: 2rem;
            }
            header {
                grid-column: 1 / -1;
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid #334155;
                padding-bottom: 1rem;
                margin-bottom: 2rem;
            }
            .panel {
                background: var(--card);
                border-radius: 12px;
                padding: 1.5rem;
                border: 1px solid #334155;
            }
            h2 { font-size: 1.25rem; font-weight: 600; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }
            .patient-card {
                background: #0f172a;
                border-radius: 8px;
                padding: 1rem;
                margin-bottom: 1rem;
                border-left: 4px solid var(--accent);
                display: flex;
                justify-content: space-between;
                align-items: center;
                animation: slideIn 0.3s ease-out;
            }
            @keyframes slideIn { from { opacity: 0; transform: translateX(-10px); } to { opacity: 1; transform: translateX(0); } }
            .vitals { font-family: 'JetBrains Mono', monospace; font-size: 0.875rem; color: var(--text-dim); }
            .badge { padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; }
            .badge-esi-1 { background: var(--danger); color: white; }
            .badge-esi-2 { background: var(--warning); color: black; }
            .badge-active { border: 1px solid var(--danger); color: var(--danger); animation: pulse 1s infinite alternate; }
            @keyframes pulse { from { opacity: 1; } to { opacity: 0.5; } }
            button {
                background: var(--accent);
                color: var(--bg);
                border: none;
                padding: 0.5rem 1rem;
                border-radius: 6px;
                font-weight: 700;
                cursor: pointer;
                transition: all 0.2s;
            }
            button:hover { filter: brightness(1.1); transform: translateY(-1px); }
            .bed-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }
            .bed-card {
                background: #0f172a;
                padding: 1rem;
                border-radius: 8px;
                text-align: center;
                border: 1px solid #334155;
            }
            .bed-count { font-size: 2rem; font-weight: 700; color: var(--accent); }
            .logs {
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.75rem;
                height: 200px;
                overflow-y: auto;
                background: #000;
                padding: 1rem;
                border-radius: 8px;
                color: #22c55e;
            }
        </style>
    </head>
    <body>
        <div class="dashboard">
            <header>
                <div>
                    <h1 style="font-size: 1.5rem; font-weight: 700;">MedTriage <span style="color: var(--accent);">Mission Control</span></h1>
                    <p style="color: var(--text-dim); font-size: 0.875rem;">Reinforcement Learning Environment Showcase</p>
                </div>
                <div style="text-align: right;">
                    <div id="clock" style="font-size: 1.5rem; font-weight: 700;">T+000</div>
                    <button onclick="resetSim()" style="background: var(--text-dim); color: white; margin-top: 0.5rem;">Reset Simulation</button>
                </div>
            </header>

            <main class="panel">
                <h2>🏥 Waiting Room</h2>
                <div id="waiting-room-list">
                    <!-- Loaded via JS -->
                </div>
            </main>

            <aside>
                <div class="panel" style="margin-bottom: 2rem;">
                    <h2>🛏️ Bed Availability</h2>
                    <div class="bed-grid">
                        <div class="bed-card">
                            <div class="bed-count" id="bed-icu">0</div>
                            <div style="font-size: 0.75rem; color: var(--text-dim);">ICU</div>
                        </div>
                        <div class="bed-card">
                            <div class="bed-count" id="bed-trauma">0</div>
                            <div style="font-size: 0.75rem; color: var(--text-dim);">Trauma</div>
                        </div>
                        <div class="bed-card">
                            <div class="bed-count" id="bed-standard">0</div>
                            <div style="font-size: 0.75rem; color: var(--text-dim);">Standard</div>
                        </div>
                    </div>
                </div>

                <div class="panel">
                    <h2>📋 Activity Logs</h2>
                    <div id="logs" class="logs">> system: simulation ready...</div>
                </div>
            </aside>
        </div>

        <script>
            async function updateState() {
                try {
                    const res = await fetch('/state');
                    const data = await res.json();
                    const state = data.state;

                    document.getElementById('clock').innerText = `T+${String(state.simulation_clock).padStart(3, '0')}`;
                    document.getElementById('bed-icu').innerText = state.bed_status.icu;
                    document.getElementById('bed-trauma').innerText = state.bed_status.trauma;
                    document.getElementById('bed-standard').innerText = state.bed_status.standard;

                    const list = document.getElementById('waiting-room-list');
                    list.innerHTML = state.waiting_room.map(p => `
                        <div class="patient-card">
                            <div>
                                <div style="font-weight: 600;">Patient ${p.patient_id} <span style="color: var(--text-dim); font-size: 0.75rem;">(Age: ${p.age})</span></div>
                                <div style="font-size: 0.875rem; margin-top: 0.25rem;">${p.complaint}</div>
                                <div class="vitals">HR: ${p.vitals.heart_rate} | RR: ${p.vitals.respiratory_rate} | SpO2: ${p.vitals.spo2}%</div>
                            </div>
                            <div style="text-align: right; display: flex; flex-direction: column; gap: 0.5rem;">
                                ${state.active_alarms.includes(p.patient_id) ? '<span class="badge badge-active">Critical Alarm</span>' : ''}
                                <div style="display: flex; gap: 0.5rem;">
                                    <button onclick="triage('${p.patient_id}', 1)" style="padding: 0.25rem 0.5rem; background: var(--danger); font-size: 0.75rem;">ESI 1</button>
                                    <button onclick="triage('${p.patient_id}', 2)" style="padding: 0.25rem 0.5rem; background: var(--warning); color: black; font-size: 0.75rem;">ESI 2</button>
                                    <button onclick="allocate('${p.patient_id}')" style="padding: 0.25rem 0.5rem; font-size: 0.75rem;">Bed</button>
                                </div>
                            </div>
                        </div>
                    `).join('');
                    
                    if (state.waiting_room.length === 0) {
                        list.innerHTML = '<div style="text-align: center; color: var(--text-dim); padding: 2rem;">Waiting room empty. Reset simulation to start again.</div>';
                    }
                } catch (e) {
                    console.error("Failed to update state", e);
                }
            }

            async function triage(patientId, level) {
                log(`Triaging ${patientId} as ESI ${level}...`);
                await fetch('/step', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({action: {action_type: 'triage_patient', patient_id: patientId, esi_level: level}})
                });
                updateState();
            }

            async function allocate(patientId) {
                log(`Attempting bed allocation for ${patientId}...`);
                const res = await fetch('/state');
                const data = await res.json();
                const beds = data.state.bed_status;
                let bedType = 'standard';
                if (beds.trauma > 0) bedType = 'trauma';
                else if (beds.icu > 0) bedType = 'icu';

                await fetch('/step', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({action: {action_type: 'allocate_bed', patient_id: patientId, bed_type: bedType}})
                });
                updateState();
            }

            async function resetSim() {
                log('Resetting simulation environment...');
                await fetch('/reset', {method: 'POST'});
                updateState();
            }

            function log(msg) {
                const logs = document.getElementById('logs');
                logs.innerHTML += `<div>> ${msg}</div>`;
                logs.scrollTop = logs.scrollHeight;
            }

            setInterval(updateState, 2000);
            updateState();
        </script>
    </body>
    </html>
    """
