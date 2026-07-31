# Must run before any other import (Flask/Werkzeug/pydantic etc. create internal threading
# primitives at import time). Under gunicorn's eventlet worker, the arbiter process imports this
# module to find the WSGI callable BEFORE the worker gets a chance to monkey-patch, and the forked
# worker inherits that already-imported, unpatched state — so patching later than this is too late
# and causes "RLock(s) were not greened" / "Working outside of request context" errors in production.
try:
    import eventlet
    eventlet.monkey_patch()
except ImportError:
    pass  # eventlet isn't installed locally (e.g. plain `python app.py` dev runs); harmless to skip

import os
import time
import math
import webbrowser
import threading
from collections import deque
import numpy as np
from dotenv import load_dotenv
from flask import Flask, send_file
from flask_socketio import SocketIO, emit
from groq import Groq

load_dotenv()

# Base directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=".", template_folder=".")
app.config["SECRET_KEY"] = "marsnet_secret_key_2026"
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Physical Constants ---
R_MARS = 3389.5
R_EARTH = 1500.0
EARTH_OFFSET_X = -35000.0

# Moons Orbit Radii (from Mars center)
R_PHOBOS = 9378.0   # ~5,989 km altitude
R_DEIMOS = 23458.0  # ~20,068 km altitude

# HEO ellipse parameters (matches frontend orbit ring geometry)
HEO_A, HEO_B, HEO_C = 16139.5, 10515.6, 12250.0

# --- Astrodynamics: Kepler's Third Law (ω ∝ r^-1.5) drives relative orbital speed ---
REF_RADIUS = R_MARS + 1000.0  # Low Mars Orbit altitude, used as the reference orbit
BASE_ANGULAR_SPEED = 1.1      # demo-timescale pace for the reference (LMO) orbit
ANGLE_STEP_COEF = 0.015       # rad per simulation tick, per unit of "speed"
TICK_DT = 0.1                 # seconds of wall-clock time per simulation tick (10 Hz)

def kepler_relative_speed(radius_km):
    """Physically-consistent relative angular speed: closer orbits move faster (Kepler's 3rd law)."""
    return BASE_ANGULAR_SPEED * (REF_RADIUS / radius_km) ** 1.5

# --- Predictive Congestion Engine tuning ---
PREDICT_TICKS = [3, 6, 9, 12, 15, 18]       # lookahead samples (up to 1.8 sim-seconds ahead)
PREDICTIVE_WARNING_DISTANCE = 1800.0        # km; larger than the reactive SAFE_DISTANCE so it fires earlier
NEARBY_PREFILTER_DISTANCE = 3000.0          # km; coarse current-distance filter before forecasting vs. other satellites

# --- Fuel / Delta-V tuning ---
DELTA_V_PER_MANEUVER = 0.5   # m/s consumed per dodge action
FUEL_COST_WEIGHT = 0.05      # reward penalty per m/s of delta-v spent (discourages needless thrusting)
DEFAULT_FUEL_CAPACITY = 50.0 # m/s mission budget per satellite

MAX_DEBRIS_COUNT = 30

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", None)
client = None
if GROQ_API_KEY:
    try:
        client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        print(f"Failed to initialize Groq Client: {e}")

#--- Reinforcement Learning Agent ---
class SatelliteRLAgent:
    """Q-Learning Agent for Autonomous Satellite Collision Avoidance & Trajectory Recovery.

    State = (dist_bin, offset_bin, predicted_bin):
      dist_bin:      0 = reactive danger (<300km), 1 = reactive warning (<1200km), 2 = clear
      offset_bin:    0 = far from home orbit (>=100km), 1 = near home orbit
      predicted_bin: 0 = predictive engine forecasts a future conflict, 1 = no forecasted conflict
    """
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.alpha = alpha # learning rate
        self.gamma = gamma #Discount factor
        self.epsilon = epsilon # Exploration rate
        self.q_table = {} # State action Q-values

        for d in [0, 1]: # Reactive danger zones (dist_bin 0 or 1)
            for o in [0, 1]:
                for p in [0, 1]:
                    #strongly favor action 1 or 2 (radial thrust dodge)
                    self.q_table[(d, o, p)] = np.array([0.0, 15.0, 15.0, -5.0])

        for o in [0, 1]: # safe zone (dist_bin 2), but predictive engine sees a conflict incoming
            self.q_table[(2, o, 0)] = np.array([2.0, 8.0, 8.0, 0.0])  # mild preemptive dodge favored

        for o in [0, 1]: # fully clear, nothing forecasted
            self.q_table[(2, o, 1)] = np.array([5.0, 0.0, 0.0, 5.0]) # favor maintaining/recovering orbit

    def get_state_key(self, min_dist, radial_offset, predicted_conflict):
        """Discretizes raw sensor distance, altitude offset & forecast into discrete states."""
        if min_dist < 300:
            dist_bin = 0
        elif min_dist < 1200:
            dist_bin = 1
        else:
            dist_bin = 2

        offset_bin = 0 if abs(radial_offset) >= 100 else 1
        predicted_bin = 0 if predicted_conflict else 1
        return (dist_bin, offset_bin, predicted_bin)

    def choose_action(self, state_key):
        '''Epsilon-greedy policy for action selection.'''
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(4)

        if np.random.uniform(0, 1) < self.epsilon:
            return int(np.random.randint(0, 4))
        else:
            return int(np.argmax(self.q_table[state_key]))

    def update(self, state_key, action, reward, next_state_key):
        '''Q-learning Bellman update equation.'''
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(4)

        predict = self.q_table[state_key][action]
        target = reward + self.gamma * np.max(self.q_table[next_state_key])
        self.q_table[state_key][action] += self.alpha * (target - predict)

rl_agent = SatelliteRLAgent()

REGIME_ALTITUDES = {
    'LMO': 1000,
    'MMO': 6000,
    'AEO': 17032,
}

def init_satellites(num_sats):
    """Generates initial orbit states and 3D positions for satellites, using Kepler-consistent speeds."""
    np.random.seed(42)
    regimes = np.random.choice(['LMO', 'MMO', 'AEO', 'HEO'], size=num_sats)
    base_angles = np.random.uniform(0, 2 * np.pi, size=num_sats)
    jitter = np.random.uniform(0.85, 1.15, size=num_sats)

    sats = []
    for i in range(num_sats):
        regime = regimes[i]
        angle = float(base_angles[i])

        if regime == 'HEO':
            radius = HEO_A
            x = float(HEO_A * np.cos(angle) + HEO_C)
            y = float(HEO_B * np.sin(angle))
        else:
            altitude = REGIME_ALTITUDES.get(regime, 1000)
            radius = R_MARS + altitude
            x = float(radius * np.cos(angle))
            y = float(radius * np.sin(angle))

        speed = float(kepler_relative_speed(radius) * jitter[i])
        z = 0.0 # Pins satellites flat on the equatorial orbital plane

        sats.append({
            "id": i + 1,
            "regime": regime,
            "radius": radius,
            "angle": angle,
            "speed": speed,
            "x": x,
            "y": y,
            "z": z,
            "radial_offset": 0.0, # offset from home orbit altitude
            "in_evasion_mode": False, # flag tracking active evasion
            "last_action": 0,
            "status": "NOMINAL",
            "predicted_conflict": False,
            "fuel_used": 0.0,
        })
    return sats

simulation_state = {
    "satellites": [],
    "debris": [],
    "num_satellites": 20,
    "orbit_altitude": 1000,
    "running": False,
    "training_mode": True,
    "debris_density": 0.3,
    "fuel_capacity": DEFAULT_FUEL_CAPACITY,
}

baseline_state = {
    "satellites": [],
}

simulation_state["satellites"] = init_satellites(simulation_state["num_satellites"])
baseline_state["satellites"] = init_satellites(simulation_state["num_satellites"])

# --- RL Avoidance Tracking Metrics ---
telemetry_data = {
    "total_reward": 0.0,
    "avoidances": 0,
    "collisions": 0,
    "baseline_collisions": 0,
}

coverage_history = deque(maxlen=1800)  # ~180s rolling window at 10Hz


def get_moon_positions(elapsed):
    """Live positions of Phobos & Deimos at a given simulation-time offset (matches frontend timing)."""
    angle_phobos = (elapsed * 0.6) % (2 * math.pi)
    angle_deimos = (elapsed * 0.15 + 1.5) % (2 * math.pi)
    return [
        (float(R_PHOBOS * math.cos(angle_phobos)), float(R_PHOBOS * math.sin(angle_phobos)), 0.0),
        (float(R_DEIMOS * math.cos(angle_deimos)), float(R_DEIMOS * math.sin(angle_deimos)), 0.0),
    ]


def project_satellite_position(sat, ticks_ahead):
    """Forward-projects a satellite's position `ticks_ahead` simulation ticks into the future."""
    angle = sat["angle"] + sat["speed"] * ANGLE_STEP_COEF * ticks_ahead
    offset = sat["radial_offset"]
    if sat["regime"] == 'HEO':
        x = (HEO_A + offset) * math.cos(angle) + HEO_C
        y = (HEO_B + offset) * math.sin(angle)
    else:
        r = sat["radius"] + offset
        x = r * math.cos(angle)
        y = r * math.sin(angle)
    return x, y, 0.0


def predict_conflicts(sat, other_sats, debris_list, elapsed_now):
    """Forecasts whether `sat` is on course to intersect a hazard within the prediction horizon."""
    nearby_sats = [
        o for o in other_sats
        if o["id"] != sat["id"] and
        (sat["x"] - o["x"]) ** 2 + (sat["y"] - o["y"]) ** 2 + (sat["z"] - o["z"]) ** 2 < NEARBY_PREFILTER_DISTANCE ** 2
    ]

    closest_dist = float("inf")
    closest_point = None

    for k in PREDICT_TICKS:
        px, py, pz = project_satellite_position(sat, k)
        future_elapsed = elapsed_now + k * TICK_DT

        hazards = [(d["x"], d["y"], d["z"]) for d in debris_list]
        hazards.extend(get_moon_positions(future_elapsed))
        hazards.extend(project_satellite_position(o, k) for o in nearby_sats)

        for hx, hy, hz in hazards:
            dist = math.sqrt((px - hx) ** 2 + (py - hy) ** 2 + (pz - hz) ** 2)
            if dist < closest_dist:
                closest_dist = dist
                closest_point = ((px + hx) / 2.0, (py + hy) / 2.0, (pz + hz) / 2.0)

    predicted_conflict = closest_dist < PREDICTIVE_WARNING_DISTANCE
    return predicted_conflict, closest_dist, closest_point


def run_simulation_step():
    """
    Checks proximity between satellites and space debris, moons (Phobos & Deimos), and other satellites,
    uses SatelliteRLAgent to pick an avoidance action, updates coordinates, and calculates rewards.
    Also advances a non-rendered "baseline" fleet (no avoidance) to measure the RL agent's collision
    reduction rate against the project's ≥70% success target.
    """
    global simulation_state, telemetry_data

    SAFE_DISTANCE = 1200.0 # Warning radius to trigger avoidance check (km)
    COLLISION_DISTANCE = 250.0 # Impact radius (km)

    elapsed = time.time()
    moons = [{"x": mx, "y": my, "z": mz} for mx, my, mz in get_moon_positions(elapsed)]

    predicted_zones = []

    for sat in simulation_state["satellites"]:
        #1. Update satellite position along orbital path
        sat["angle"] += sat["speed"] * ANGLE_STEP_COEF

        if sat["regime"] == 'HEO':
            offset = sat["radial_offset"]
            sat["x"] = float((HEO_A + offset) * np.cos(sat["angle"]) + HEO_C)
            sat["y"] = float((HEO_B + offset) * np.sin(sat["angle"]))
        else:
            current_radius = sat["radius"] + sat["radial_offset"]
            sat["x"] = float(current_radius * np.cos(sat["angle"]))
            sat["y"] = float(current_radius * np.sin(sat["angle"]))

        #2. Measure distance to closest hazard (space debris object, moons, or other satellites)
        min_dist = 999999.0

        for debris in simulation_state["debris"]:
            dist = math.sqrt(
                (sat["x"] - debris["x"]) ** 2 +
                (sat["y"] - debris["y"]) ** 2 +
                (sat["z"] - debris["z"]) ** 2
            )
            if dist < min_dist:
                min_dist = dist

        for moon in moons:
            dist = math.sqrt(
                (sat["x"] - moon["x"]) ** 2 +
                (sat["y"] - moon["y"]) ** 2 +
                (sat["z"] - moon["z"]) ** 2
            )
            if dist < min_dist:
                min_dist = dist

        for other_sat in simulation_state["satellites"]:
            if other_sat["id"] != sat["id"]:
                dist = math.sqrt(
                    (sat["x"] - other_sat["x"]) ** 2 +
                    (sat["y"] - other_sat["y"]) ** 2 +
                    (sat["z"] - other_sat["z"]) ** 2
                )
                if dist < 400.0 and dist < min_dist:
                    min_dist = dist

        #3. Predictive Congestion Engine: forecast conflicts before they become reactive threats
        predicted_conflict, _, predicted_point = predict_conflicts(
            sat, simulation_state["satellites"], simulation_state["debris"], elapsed
        )
        sat["predicted_conflict"] = predicted_conflict
        if predicted_conflict and predicted_point:
            predicted_zones.append({"x": predicted_point[0], "y": predicted_point[1], "z": predicted_point[2]})

        #4. RL Agent State & Action Selection
        remaining_fuel = simulation_state["fuel_capacity"] - sat["fuel_used"]
        out_of_fuel = remaining_fuel < DELTA_V_PER_MANEUVER

        # State captured BEFORE the action is applied (radial_offset still pre-maneuver)
        state_key = rl_agent.get_state_key(min_dist, sat["radial_offset"], predicted_conflict)

        if not simulation_state["training_mode"]:
            # Training mode off: fleet flies the naive baseline policy for live A/B comparison
            action = 0
        elif min_dist < 400.0:
            #Force opposite directions based on satellite ID so they don't crash into each other
            action = 1 if sat["id"] % 2 == 0 else 2
            sat["in_evasion_mode"] = True
        else:
            action = rl_agent.choose_action(state_key)

        if action in (1, 2) and out_of_fuel:
            action = 0 # No fuel left for a maneuver; fall back to holding position

        sat["last_action"] = action

        # Action 0: Maintain orbit
        # Action 1: Radial thrust outward (+25 km altitude)
        # Action 2: Radial thrust inward (-25 km altitude)
        # Action 3: Restore nominal orbit altitude
        reward = 0.1 # Routine orbit reward

        if action == 1: # Dodge outward
            sat["radial_offset"] += 25.0
            sat["in_evasion_mode"] = True
            sat["fuel_used"] += DELTA_V_PER_MANEUVER
            reward -= FUEL_COST_WEIGHT * DELTA_V_PER_MANEUVER
        elif action == 2: # Dodge Inward
            sat["radial_offset"] -= 25.0
            sat["in_evasion_mode"] = True
            sat["fuel_used"] += DELTA_V_PER_MANEUVER
            reward -= FUEL_COST_WEIGHT * DELTA_V_PER_MANEUVER
        elif action == 3 or not sat["in_evasion_mode"]: # Recover back to home orbit
            sat["radial_offset"] *= 0.70
            if abs(sat["radial_offset"]) < 2.0:
                sat["radial_offset"] = 0.0
                sat["in_evasion_mode"] = False

        #if the satellite is actively evading or facing a collision threat, let it push out far (up to 800 km).
        # otherwise gently pull it back and lock it tightly to its designated orbit ring (+- 100 km)
        if sat["status"] in ["EVADING", "COLLISION"]:
            sat["radial_offset"] = max(-800.0, min(800.0, sat["radial_offset"]))
        else:
            sat["radial_offset"] = max(-100.0, min(100.0, sat["radial_offset"]))

        #5. Collision vs Avoidance Evaluation & rewards
        if min_dist < COLLISION_DISTANCE:
            telemetry_data["collisions"] += 1
            reward = -25.0 # higher penalty for actual impacts
            sat["status"] = "COLLISION"
        elif min_dist < SAFE_DISTANCE:
            if sat["in_evasion_mode"] or action in [1, 2]:
                #only count an avoidance once per evasion maneuver state transition to prevent counter inflation
                if sat["status"] != "EVADING":
                    telemetry_data["avoidances"] += 1
                reward = 15.0 # positive reward for successful evasion
                sat["status"] = "EVADING"
        elif predicted_conflict:
            sat["status"] = "PREDICTED_CONFLICT"
            sat["in_evasion_mode"] = False
        elif out_of_fuel:
            sat["status"] = "NO_FUEL"
            sat["in_evasion_mode"] = False
        else:
            reward = 1.0
            sat["status"] = "NOMINAL"
            sat["in_evasion_mode"] = False

        # Target Goal Shaping: Enforce 99% avoidance success shaping reward
        total_encounters = telemetry_data["avoidances"] + telemetry_data["collisions"]
        if total_encounters > 0:
            current_avoidance_rate = telemetry_data["avoidances"] / total_encounters
            if current_avoidance_rate < 0.99:
                reward -= 3.0 # Penalty if agent falls below the 99% avoidance target

        #6. Update RL Agent Q-table (skipped while training mode is off)
        if simulation_state["training_mode"]:
            # next_state reflects the post-maneuver radial_offset
            next_state_key = rl_agent.get_state_key(min_dist, sat["radial_offset"], predicted_conflict)
            rl_agent.update(state_key, action, reward, next_state_key)
        telemetry_data["total_reward"] += reward

    simulation_state["predicted_zones"] = predicted_zones

    #--- Baseline (no-avoidance) shadow fleet: measures the RL agent's collision reduction rate ---
    for bsat in baseline_state["satellites"]:
        bsat["angle"] += bsat["speed"] * ANGLE_STEP_COEF
        if bsat["regime"] == 'HEO':
            bsat["x"] = float(HEO_A * np.cos(bsat["angle"]) + HEO_C)
            bsat["y"] = float(HEO_B * np.sin(bsat["angle"]))
        else:
            bsat["x"] = float(bsat["radius"] * np.cos(bsat["angle"]))
            bsat["y"] = float(bsat["radius"] * np.sin(bsat["angle"]))

        min_dist = 999999.0
        for debris in simulation_state["debris"]:
            dist = math.sqrt(
                (bsat["x"] - debris["x"]) ** 2 + (bsat["y"] - debris["y"]) ** 2 + (bsat["z"] - debris["z"]) ** 2
            )
            if dist < min_dist:
                min_dist = dist
        for moon in moons:
            dist = math.sqrt(
                (bsat["x"] - moon["x"]) ** 2 + (bsat["y"] - moon["y"]) ** 2 + (bsat["z"] - moon["z"]) ** 2
            )
            if dist < min_dist:
                min_dist = dist
        for other in baseline_state["satellites"]:
            if other["id"] != bsat["id"]:
                dist = math.sqrt(
                    (bsat["x"] - other["x"]) ** 2 + (bsat["y"] - other["y"]) ** 2 + (bsat["z"] - other["z"]) ** 2
                )
                if dist < 400.0 and dist < min_dist:
                    min_dist = dist

        if min_dist < COLLISION_DISTANCE:
            telemetry_data["baseline_collisions"] += 1


def background_simulation_loop():
    """Continuously runs collision avoidance checks and pushes updates to frontend."""
    print("[MarsNet] background simulation loop starting", flush=True)
    tick_count = 0
    while True:
        socketio.sleep(TICK_DT)

        try:
            if not simulation_state["running"]:
                continue

            # Debris density slider: probabilistically auto-spawns debris while the sim runs
            density = simulation_state["debris_density"]
            if density > 0 and np.random.random() < density * 0.02:
                spawn_debris()

            run_simulation_step()
            _emit_telemetry()

            tick_count += 1
            if tick_count % 100 == 0:  # heartbeat roughly every 10s at 10Hz, so Render logs prove it's alive
                print(f"[MarsNet] heartbeat: tick={tick_count} running={simulation_state['running']} "
                      f"satellites={len(simulation_state['satellites'])}", flush=True)
        except Exception:
            # A single bad tick must never permanently kill the loop (this previously happened
            # silently — no more telemetry_update events would ever fire again, with nothing
            # visible to the user beyond "the simulation just stopped moving").
            import traceback
            print("[MarsNet] ERROR in background_simulation_loop tick — continuing:", flush=True)
            traceback.print_exc()


def _emit_telemetry():
    total_encounters = telemetry_data["avoidances"] + telemetry_data["collisions"]
    success_rate = (telemetry_data["avoidances"] / total_encounters * 100.0) if total_encounters > 0 else 100.0

    active_functional_sats = sum(1 for s in simulation_state["satellites"] if s["status"] in ["NOMINAL", "EVADING", "PREDICTED_CONFLICT"])
    total_sats = len(simulation_state["satellites"])
    network_coverage = (active_functional_sats / total_sats * 100.0) if total_sats else 100.0

    active_threats = sum(1 for s in simulation_state["satellites"] if s["status"] == "COLLISION")
    collision_risk = min(100.0, (active_threats / max(1, total_sats)) * 100.0)
    life_risk = collision_risk * 0.5

    coverage_history.append(1 if network_coverage >= 90.0 else 0)
    coverage_uptime_pct = (sum(coverage_history) / len(coverage_history) * 100.0) if coverage_history else 100.0

    rl_collisions = telemetry_data["collisions"]
    baseline_collisions = telemetry_data["baseline_collisions"]
    collision_reduction_rate = (
        (1 - rl_collisions / baseline_collisions) * 100.0 if baseline_collisions > 0 else None
    )

    fuel_capacity = simulation_state["fuel_capacity"]
    if total_sats:
        avg_fuel_remaining_pct = sum(
            max(0.0, (fuel_capacity - s["fuel_used"]) / fuel_capacity) for s in simulation_state["satellites"]
        ) / total_sats * 100.0
    else:
        avg_fuel_remaining_pct = 100.0

    predicted_conflicts_count = sum(1 for s in simulation_state["satellites"] if s["predicted_conflict"])

    socketio.emit("telemetry_update", {
        "total_reward": round(telemetry_data["total_reward"], 1),
        "avoidances": int(telemetry_data["avoidances"]),
        "collisions": int(rl_collisions),
        "baseline_collisions": int(baseline_collisions),
        "collision_reduction_rate": round(collision_reduction_rate, 1) if collision_reduction_rate is not None else None,
        "success_rate": round(success_rate, 1),
        "network_coverage": round(network_coverage, 1),
        "coverage_uptime_pct": round(coverage_uptime_pct, 1),
        "collision_risk": round(collision_risk, 1),
        "life_risk": round(life_risk, 1),
        "fleet_avg_fuel_remaining_pct": round(avg_fuel_remaining_pct, 1),
        "predicted_conflicts_count": int(predicted_conflicts_count),
        "predicted_zones": simulation_state.get("predicted_zones", []),
        "satellites": simulation_state["satellites"],
    })


def spawn_debris():
    # At capacity, retire the oldest debris instead of silently refusing — keeps the
    # "Inject Chaos" button (and density-driven auto-spawn) always visibly responsive.
    if len(simulation_state["debris"]) >= MAX_DEBRIS_COUNT:
        simulation_state["debris"].pop(0)

    r_orbit = R_MARS + simulation_state["orbit_altitude"]
    angle = np.random.uniform(0, 2 * np.pi)
    debris_item = {
        "x": float(r_orbit * np.cos(angle)),
        "y": float(r_orbit * np.sin(angle)),
        "z": float(np.random.uniform(-400, 400))
    }
    simulation_state["debris"].append(debris_item)
    socketio.emit("debris_updated", {"count": len(simulation_state["debris"]), "debris": simulation_state["debris"]})


def reset_all_state():
    """Resets the scenario (satellites, debris, telemetry, fuel) but keeps the RL agent's learned Q-table."""
    num_sats = simulation_state["num_satellites"]
    simulation_state["satellites"] = init_satellites(num_sats)
    baseline_state["satellites"] = init_satellites(num_sats)
    simulation_state["debris"] = []
    simulation_state["predicted_zones"] = []
    telemetry_data["total_reward"] = 0.0
    telemetry_data["avoidances"] = 0
    telemetry_data["collisions"] = 0
    telemetry_data["baseline_collisions"] = 0
    coverage_history.clear()


# --- Web Routes ---
@app.route("/")
def serve_landing():
    """Serves the landing page with an explicit path."""
    return send_file(os.path.join(BASE_DIR, "landing.html"))

@app.route("/app")
def serve_dashboard():
    """Serves the 3D MarsNet-RL simulation dashboard with an explicit path."""
    return send_file(os.path.join(BASE_DIR, "index.html"))

# --- Socket.IO Events ---
@socketio.on("connect")
def handle_connect():
    print(f"[MarsNet] client connected - server running={simulation_state['running']}", flush=True)
    emit("init_state", {
        "num_satellites": simulation_state["num_satellites"],
        "satellites": simulation_state["satellites"],
        "debris": simulation_state["debris"],
        "running": simulation_state["running"],
        "training_mode": simulation_state["training_mode"],
        "debris_density": simulation_state["debris_density"],
        "fuel_capacity": simulation_state["fuel_capacity"],
    })

@socketio.on("update_satellites")
def handle_update_satellites(data):
    num_sats = int(data.get("count", 20))
    simulation_state["num_satellites"] = num_sats
    simulation_state["satellites"] = init_satellites(num_sats)
    baseline_state["satellites"] = init_satellites(num_sats)
    emit("satellites_updated", {"satellites": simulation_state["satellites"]}, broadcast=True)

@socketio.on("inject_chaos")
def handle_inject_chaos():
    spawn_debris()

@socketio.on("update_target_orbit")
def handle_update_target_orbit(data):
    simulation_state["orbit_altitude"] = max(200.0, float(data.get("altitude", 1000)))

@socketio.on("start_simulation")
def handle_start_simulation():
    simulation_state["running"] = True
    print("[MarsNet] start_simulation received - running=True", flush=True)
    emit("simulation_status", {"running": True}, broadcast=True)

@socketio.on("pause_simulation")
def handle_pause_simulation():
    simulation_state["running"] = False
    print("[MarsNet] pause_simulation received - running=False", flush=True)
    emit("simulation_status", {"running": False}, broadcast=True)

@socketio.on("reset_simulation")
def handle_reset_simulation():
    print("[MarsNet] reset_simulation received", flush=True)
    reset_all_state()
    emit("simulation_status", {"running": simulation_state["running"]}, broadcast=True)
    emit("init_state", {
        "num_satellites": simulation_state["num_satellites"],
        "satellites": simulation_state["satellites"],
        "debris": simulation_state["debris"],
        "running": simulation_state["running"],
        "training_mode": simulation_state["training_mode"],
        "debris_density": simulation_state["debris_density"],
        "fuel_capacity": simulation_state["fuel_capacity"],
    }, broadcast=True)

@socketio.on("update_debris_density")
def handle_update_debris_density(data):
    simulation_state["debris_density"] = max(0.0, min(1.0, float(data.get("density", 0.3))))

@socketio.on("toggle_training_mode")
def handle_toggle_training_mode(data):
    simulation_state["training_mode"] = bool(data.get("enabled", True))

@socketio.on("update_fuel_capacity")
def handle_update_fuel_capacity(data):
    simulation_state["fuel_capacity"] = max(1.0, float(data.get("capacity", DEFAULT_FUEL_CAPACITY)))

@socketio.on("send_message")
def handle_chat_message(data):
    user_message = data.get("message", "")
    history = data.get("history", [])

    if not client:
        emit("chat_response_chunk", {"chunk": "Error: Groq API key is missing."})
        emit("chat_response_end")
        return

    total_encounters = telemetry_data["avoidances"] + telemetry_data["collisions"]
    reduction = (
        f"{(1 - telemetry_data['collisions'] / telemetry_data['baseline_collisions']) * 100:.1f}%"
        if telemetry_data["baseline_collisions"] > 0 else "not yet measurable (baseline fleet hasn't collided yet)"
    )
    live_context = (
        "Live simulation state you must ground every answer in (do not invent satellite names, missions, or "
        "events that aren't reflected here):\n"
        f"- Simulation is currently {'RUNNING' if simulation_state['running'] else 'PAUSED'}.\n"
        f"- Fleet size: {len(simulation_state['satellites'])} autonomous satellites across LMO/MMO/AEO/HEO Mars orbits.\n"
        f"- Space debris objects currently tracked: {len(simulation_state['debris'])}.\n"
        f"- Cumulative RL reward: {telemetry_data['total_reward']:.1f}.\n"
        f"- Reactive avoidances: {telemetry_data['avoidances']}, collisions: {telemetry_data['collisions']}, "
        f"baseline (no-RL) collisions: {telemetry_data['baseline_collisions']}.\n"
        f"- Collision reduction rate vs. no-RL baseline: {reduction} (target is >=70%).\n"
        f"- Training mode (RL learning): {'ON' if simulation_state['training_mode'] else 'OFF (flying naive baseline policy)'}.\n"
        f"- Fuel budget per satellite: {simulation_state['fuel_capacity']:.1f} m/s Delta-V.\n"
    )

    messages = [{
        "role": "system",
        "content": (
            "You are MarsNet Assistant, a mission control copilot for the MarsNet-RL autonomous satellite "
            "simulation — a demonstrator of reinforcement-learning-driven collision avoidance for a Mars "
            "satellite constellation, built because the ~20-minute Earth-Mars signal delay makes real-time "
            "ground control impossible. Answer questions about the simulation, the RL/predictive-avoidance "
            "approach, and orbital mechanics concisely and factually. Use the live state below rather than "
            "fabricating scenario details.\n\n" + live_context
        )
    }] + history + [{"role": "user", "content": user_message}]

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            stream=True
        )
        for chunk in completion:
            content = chunk.choices[0].delta.content
            if content:
                emit("chat_response_chunk", {"chunk": content})
        emit("chat_response_end")
    except Exception as e:
        emit("chat_response_chunk", {"chunk": f"Error: {str(e)}"})
        emit("chat_response_end")

threading.Thread(target=background_simulation_loop, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    if not os.environ.get("RENDER"):
        def open_browser():
            time.sleep(1.5)
            webbrowser.open(f"http://127.0.0.1:{port}")

        threading.Thread(target=open_browser, daemon=True).start()

    socketio.run(app, debug=False, host="0.0.0.0", port=port, allow_unsafe_werkzeug=True)
