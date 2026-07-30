import os
import time
import math
import webbrowser
import threading
import numpy as np
from flask import Flask, send_file
from flask_socketio import SocketIO, emit
from groq import Groq

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

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", None)
client = None
if GROQ_API_KEY:
    try:
        client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        print(f"Failed to initialize Groq Client: {e}")

#--- Reinforcement Learning Agent ---
class SatelliteRLAgent:
    """Q-Learning Agent for Autonomous Satellite Collision Avoidance & Trajectory Recovery."""
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.alpha = alpha # learning rate
        self.gamma = gamma #Discount factor
        self.epsilon = epsilon # Exploration rate
        self.q_table = {} # State action Q-values

        for d in [0,1]: # Danger zones(dist_bin 0 or 1)
            for o in [0,1]: # offset bins
                #strongly favor action 1 or 2 (radial thrust doge)
                self.q_table[(d,o)] = np.array([0.0, 15.0, 15.0, -5.0])

        for o in [0,1]: # safe zone (dist_bin 2)
            # favor maintaining orbit (action 0) or recovering (action 3)
            self.q_table[(2, o)] = np.array([5.0, 0.0, 0.0, 5.0])

    def get_state_key(self, min_dist, radial_offset):
        """Discretizes raw sensor distance & altitude offset into discrete states."""
        if min_dist < 300:
            dist_bin = 0
        elif min_dist < 1200:
            dist_bin = 1
        else:
                dist_bin = 2

        offset_bin = 0 if abs(radial_offset) >= 100 else 1
        return (dist_bin, offset_bin)

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
                       
    

simulation_state = {
    "satellites": [],
    "debris": [],
    "num_satellites": 20,
    "orbit_altitude": 1000
}

def init_satellites(num_sats):
    """Generates initial orbit states and 3D positions for satellites."""
    np.random.seed(42)
    regimes = np.random.choice(['LMO', 'MMO', 'AEO', 'HEO'], size=num_sats)
    base_angles = np.random.uniform(0, 2 * np.pi, size=num_sats)
    speeds = np.random.uniform(0.3, 2.0, size=num_sats)
    
    regime_altitudes = {
        'LMO': 1000,
        'MMO': 6000,
        'AEO': 17032
    }
    
    sats = []
    for i in range(num_sats):
        regime = regimes[i]
        angle = float(base_angles[i])

        if regime == 'HEO':
            #Elliptical parameters matching index.html
            a, b, c = 16139.5, 10515.6, 12250.0
            radius = a
            x = float(a * np.cos(angle) + c)
            y = float(b * np.sin(angle))
        else:
            altitude = regime_altitudes.get(regime, 1000)
            radius = R_MARS + altitude
            x = float(radius * np.cos(angle))
            y = float(radius * np.sin(angle))

        z = 0.0 # Pins sayellites flat on the equatorial orbital plane

        sats.append({
            "id": i + 1,
            "regime": regime,
            "radius": radius,
            "angle": angle,
            "speed": float(speeds[i]),
            "x": x,
            "y": y,
            "z": z,
            "radial_offset": 0.0, # offset from home orbit altitude
            "in_evasion_mode": False, # flag tracking active evasion
            "last_action": 0,
            "status": "NOMINAL"
        })
    return sats

simulation_state["satellites"] = init_satellites(simulation_state["num_satellites"])

# --- RL Avoidance Tracking Metrics ---
telemetry_data = {
    "total_reward": 0.0,
    "avoidances": 0,
    "collisions": 0
}

def run_simulation_step():
    """
    Checks proximity between satellites and space debris, moons (Phobos & Deimos), and other satellites, 
    uses SatelliteRLAgent to pick an avoidance action, updates coordinates, and calculates rewards.
    """
    global simulation_state, telemetry_data

    SAFE_DISTANCE = 1200.0 # Warning radius to trigger avoidance check (km)
    COLLISION_DISTANCE = 250.0 # Impact radius (km)

    # Calculate live dynamic poitions of phobos and Deimos matching frontend timing
    elapsed = time.time()
    angle_phobos = (elapsed * 0.6) % (2 * math.pi)
    angle_deimos = (elapsed * 0.15 + 1.5) % (2 * math.pi)

    moons = [
        {"x": float(R_PHOBOS * math.cos(angle_phobos)), "y": float(R_PHOBOS * math.sin(angle_phobos)), "z":0.0},
        {"x": float(R_DEIMOS * math.cos(angle_deimos)), "y": float(R_DEIMOS * math.sin(angle_deimos)), "z":0.0},
    ]

    for sat in simulation_state["satellites"]:
        #1. U[date satellites posiiton along orbital path
        sat["angle"] += sat["speed"] * 0.015

        if sat["regime"] == 'HEO':
            a, b, c = 16139.5, 10515.6, 12250.0
            offset = sat["radial_offset"]
            sat["x"] = float((a + offset) * np.cos(sat["angle"]) + c)
            sat["y"] = float((b + offset) * np.sin(sat["angle"]))
        else:
            current_radius = sat["radius"] + sat["radial_offset"]
            sat["x"] = float(current_radius * np.cos(sat["angle"]))
            sat["y"] = float(current_radius * np.sin(sat["angle"]))

        #2. Measure distance to closest hazard (space debris object, moons, or pther satellites)
        min_dist = 999999.0

        # chack against space debris
        for debris in simulation_state["debris"]:
            dist = math.sqrt(
                (sat["x"] - debris["x"]) ** 2 +
                (sat["y"] - debris["y"]) ** 2 +
                (sat["z"] - debris["z"]) ** 2 
            )
            if dist < min_dist:
                min_dist = dist

        # Check against moons (Phobos & Deimos)
        for moon in moons:
            dist = math.sqrt(
                (sat["x"] - moon["x"]) ** 2 +
                (sat["y"] - moon["y"]) ** 2 +
                (sat["z"] - moon["z"]) ** 2 
            )
            if dist < min_dist:
                min_dist = dist

        # check against all other satellites, BUT only track them if they get critically sclose (<400 km)
        for other_sat in simulation_state["satellites"]:
            if other_sat["id"] != sat["id"]:
                dist = math.sqrt(
                    (sat["x"] - other_sat["x"]) ** 2 +
                    (sat["y"] - other_sat["y"]) ** 2 +
                    (sat["z"] - other_sat["z"]) ** 2 
                )
                if dist < 400.0 and dist < min_dist:
                    min_dist = dist

        #3. RL Agent State & Action Selection
        state_key = rl_agent.get_state_key(min_dist, sat["radial_offset"])

        if min_dist < 400.0:
            #Force opposite directions base don satellites ID so they don't crahs into each other
            action = 1 if sat ["id"] % 2 == 0 else 2
            sat["in_evasion_mode"] = True
        else: 
            action = rl_agent.choose_action(state_key)

       
        sat["last_action"] = action 

        # Action 0: Maintain orbit
        # Action 1: Radial thrust outward (+25 km altitude)
        # Action 2: Radial thrust inward (-km altitude)
        # Action 3: Restore nominal orbit altitude
        reward = 0.1 # Routine orbit reward

        if action == 1: # Dodge outward
            sat["radial_offset"] += 25.0
            sat["in_evasion_mode"] = True
        elif action == 2: # Dodge Inward
            sat["radial_offset"] -= 25.0
            sat["in_evasion_mode"] = True
        elif action == 3 or not sat["in_evasion_mode"]: # Recover back to home orbit
            sat["radial_offset"] *= 0.70
            if abs(sat["radial_offset"]) < 2.0:
                sat["radial_offset"] = 0.0
                sat["in_evasion_mode"] = False

        #if the saetllites is activly evading or facing a collision threat, let it push out far(up to 800 km).
        # otherwise gently pull it back and lock it tighlty to its designated orbit ring (+- 100 km)
        if sat["status"] in ["EVADING", "COLLISION"]:
            sat["radial_offset"] = max(-800.0, min(800.0, sat["radial_offset"]))
        else:
            sat["radial_offset"] = max(-100.0, min(100.0, sat["radial_offset"]))
        #4. Collision vs Avoidance Evaluation & rewards

        if min_dist < COLLISION_DISTANCE:
            telemetry_data["collisions"] += 1
            reward = -25.0 # higher penalty for actial impacts
            sat["status"] = "COLLISION"
        elif min_dist < SAFE_DISTANCE:
            if sat["in_evasion_mode"] or action in [1,2]:
                #only cound an avoidanc eonce per evasion maneuver state transition to rpevent counter inflation
                if sat["status"] != "EVADING":
                    telemetry_data["avoidances"] += 1
                reward = 15.0 # positive reward for sucessful evasion
                sat["status"] = "EVADING"
        else:
            reward = 1.0 # penalty for 
            sat["status"] = "NOMINAL"
            sat["in_evasion_mode"] = False
            

        # Target Goal Shaping: Enforce 99% avoidance sucess shaping reward
        total_encounters = telemetry_data["avoidances"] + telemetry_data["collisions"]
        if total_encounters > 0:
            current_avoidance_rate = telemetry_data["avoidances"] / total_encounters
            if current_avoidance_rate < 0.99:
                reward -= 3.0 # Penalty if agent falls below the 99% avoidance target

        #5. Update RL Agent Q-table
        next_state_key = rl_agent.get_state_key(min_dist, sat["radial_offset"])
        rl_agent.update(state_key, action, reward, next_state_key)
        telemetry_data["total_reward"] += reward


def background_simulation_loop():
    """Continuously runs collision avoidance checks and pushes updates tp frontend."""
    while True:
        socketio.sleep(0.1) # Runs at 10 Hz
        run_simulation_step()

        total_encounters = telemetry_data["avoidances"] + telemetry_data["collisions"]

        #1. sucess rate: default to 100% if no encounters yet, avoidances/ (avoidances + collisions), default to 100% if no hazards yet
        if total_encounters > 0:
            success_rate = (telemetry_data ["avoidances"] / total_encounters) * 100.0
        else: 
            success_rate =  100.0 #start healthy at 100% when no incedents ahve occured

        # 2. Network Coverage: Percentage of satelites in 'NOMINAL'
        active_functional_sats = sum(1 for s in simulation_state["satellites"] if s["status"] in ["NOMINAL", "EVADING"])
        network_coverage = (active_functional_sats / len(simulation_state["satellites"]) * 100.0) if simulation_state["satellites"] else 100.0
        #3. Collision Rist and life risk based on active close proximity threats
        active_threats = sum(1 for s in simulation_state["satellites"] if s["status"] == "COLLISION") 
        collision_risk = min(100.0, (active_threats/max(1, len(simulation_state["satellites"])))*100.0)
        life_risk = collision_risk * 0.5 # Scaled safety risk factor

        # Sends calculated metric to index.html
        socketio.emit("telemetry_update", {
            "total_reward": round(telemetry_data["total_reward"], 1),
            "avoidances": int(telemetry_data["avoidances"]),
            "collisions": int(telemetry_data["collisions"]),
            "success_rate": round(success_rate, 1),
            "network_coverage": round(network_coverage, 1),
            "collision_risk": round(collision_risk, 1),
            "life_risk": round(life_risk, 1),
            "satellites": simulation_state["satellites"],
        })


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
    emit("init_state", {
        "num_satellites": simulation_state["num_satellites"],
        "satellites": simulation_state["satellites"],
        "debris": simulation_state["debris"]
    })

@socketio.on("init_state_sync")
def handle_init_state_sync():
    emit("init_state", {
        "num_satellites": simulation_state["num_satellites"],
        "satellites": simulation_state["satellites"],
        "debris": simulation_state["debris"]
    })

@socketio.on("update_satellites")
def handle_update_satellites(data):
    num_sats = int(data.get("count", 20))
    simulation_state["num_satellites"] = num_sats
    simulation_state["satellites"] = init_satellites(num_sats)
    emit("satellites_updated", {"satellites": simulation_state["satellites"]}, broadcast=True)

@socketio.on("inject_chaos")
def handle_inject_chaos():
    r_orbit = R_MARS + simulation_state["orbit_altitude"]
    if len(simulation_state["debris"]) < 20:
        angle = np.random.uniform(0, 2 * np.pi)
        debris_item = {
            "x": float(r_orbit * np.cos(angle)),
            "y": float(r_orbit * np.sin(angle)),
            "z": float(np.random.uniform(-400, 400))
        }
        simulation_state["debris"].append(debris_item)
        emit("debris_updated", {"count": len(simulation_state["debris"])}, broadcast=True)

@socketio.on("send_message")
def handle_chat_message(data):
    user_message = data.get("message", "")
    history = data.get("history", [])
    
    if not client:
        emit("chat_response_chunk", {"chunk": "Error: Groq API key is missing."})
        emit("chat_response_end")
        return

    messages = [{
        "role": "system",
        "content": "You are MarsNet Assistant, a mission control copilot for autonomous satellite operations."
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

if __name__ == "__main__":
    def open_browser():
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:5000")

    threading.Thread(target=open_browser, daemon=True).start()
    threading.Thread(target=background_simulation_loop, daemon=True).start()
    socketio.run(app, debug=True, port=5000, allow_unsafe_werkzeug=True)