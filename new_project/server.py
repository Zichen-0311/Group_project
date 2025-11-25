from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import numpy as np
import os
import matplotlib.pyplot as plt

from generate_sequence_trajectory import generate_sequence_trajectory
from trajectory_classifier import compute_metrics, score_ai, plot_trajectory

app = Flask(__name__)
CORS(app)


# ======================================================
# Generate four math problems, each answer is a single digit (0–9)
# ======================================================
def generate_math_challenge():
    problems = []
    results = []  # per-question answers as strings "0"–"9"

    ops = ["+", "-", "*"]

    for i in range(4):
        while True:
            a = random.randint(1, 20)
            b = random.randint(1, 20)
            op = random.choice(ops)

            if op == "+":
                ans = a + b
            elif op == "-":
                ans = a - b
            else:
                ans = a * b

            # only accept single digit answers
            if 0 <= ans <= 9:
                # question text with index
                problems.append(f"({i+1})  {a} {op} {b}")
                results.append(str(ans))
                break

    return problems, results


# ======================================================
# Provide math problems + shuffled order + true code
# ======================================================
@app.route("/captcha")
def get_captcha():
    # Generate four problems and their answers
    problems, results = generate_math_challenge()  # results is ["a1","a2","a3","a4"]

    # Shuffle order of question indices "1".."4"
    order = ["1", "2", "3", "4"]
    random.shuffle(order)

    # Build final code according to shuffled order
    # e.g. results = ["5","0","7","3"], order = ["3","1","4","2"]
    # code = results[2] + results[0] + results[3] + results[1]
    code = "".join(results[int(idx) - 1] for idx in order)

    # Print debug info to the terminal
    print("=== New CAPTCHA Generated ===")
    print("Problems:")
    for i, p in enumerate(problems, start=1):
        print(f"  Q{i}: {p}")
    print("Answers per question (Q1–Q4):", results)
    print("Order used for code:", order)
    print("Final code (user must type):", code)
    print("=============================")

    return jsonify({
        "questions": problems,  # list of 4 strings "(1) a + b"
        "order": order,         # e.g. ["3","1","4","2"], shown to user
        "code": code            # not shown on page, only used for checking
    })


# ======================================================
# Normalize points to numpy array
# ======================================================
def normalize_points(lst):
    """Convert point list from frontend into Nx2 numpy array."""
    pts = []
    for p in lst:
        if isinstance(p, dict):
            pts.append([float(p["x"]), float(p["y"])])
        else:
            pts.append([float(p[0]), float(p[1])])
    return np.array(pts, dtype=float)


# ======================================================
# Human verification: check answer + trajectory
# ======================================================
@app.route("/human_attack", methods=["POST"])
def human_attack():
    data = request.get_json(force=True)
    true_code = data.get("sequence", "")   # true code from backend, e.g. "5073"
    answer = data.get("answer", "")        # user input string from textbox
    raw_points = data.get("points", [])

    # First: check answer correctness
    if answer != true_code:
        return jsonify({
            "success": False,
            "msg": "Answer incorrect.",
            "ai_prob": -1.0  # -1 means not evaluated
        })

    # Answer correct -> now check trajectory
    points = normalize_points(raw_points)
    if len(points) < 8:
        return jsonify({
            "success": False,
            "msg": "Answer correct, but trajectory too short to verify.",
            "ai_prob": 1.0
        })

    speed, accel, curv = compute_metrics(points.tolist())
    ai_prob = float(score_ai(speed, accel, curv))

    # If trajectory looks like AI, reject
    if ai_prob >= 0.5:
        return jsonify({
            "success": False,
            "msg": "Answer correct, but trajectory classified as AI. Access denied.",
            "ai_prob": ai_prob
        })
    else:
        return jsonify({
            "success": True,
            "msg": "Answer correct and trajectory looks human. Access granted.",
            "ai_prob": ai_prob
        })


# ======================================================
# AI attack: generate AI trajectory using ML model
# ======================================================
@app.route("/ai_attack", methods=["POST"])
def ai_attack():
    data = request.get_json(force=True)
    seq = data.get("sequence", "")  # same code used for keypad, e.g. "5073"

    traj = generate_sequence_trajectory(seq, save_json=False)
    speed, accel, curv = compute_metrics(traj.tolist())
    ai_prob = float(score_ai(speed, accel, curv))

    return jsonify({
        "success": True,
        "ai_prob": ai_prob,
        "points": traj.tolist()
    })


# ======================================================
# Analyze trajectories and save images (human vs AI)
# ======================================================
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)

    session = data.get("session_id", "default")
    human_pts = normalize_points(data.get("human_points", []))
    ai_pts = normalize_points(data.get("ai_points", []))

    if human_pts.size == 0 or ai_pts.size == 0:
        return jsonify({"success": False, "msg": "Trajectory data missing."})

    # Compute metrics
    h_speed, h_acc, h_cur = compute_metrics(human_pts.tolist())
    a_speed, a_acc, a_cur = compute_metrics(ai_pts.tolist())

    hs = float(score_ai(h_speed, h_acc, h_cur))
    as_ = float(score_ai(a_speed, a_acc, a_cur))

    # Save trajectory metric plots
    plot_trajectory("Human", h_speed, h_acc, h_cur)
    plot_trajectory("AI", a_speed, a_acc, a_cur)

    # Save raw trajectories
    os.makedirs("static/trajectory_images", exist_ok=True)
    h_img = f"static/trajectory_images/{session}_human.png"
    a_img = f"static/trajectory_images/{session}_ai.png"

    plt.plot(human_pts[:, 0], human_pts[:, 1])
    plt.gca().invert_yaxis()
    plt.savefig(h_img)
    plt.close()

    plt.plot(ai_pts[:, 0], ai_pts[:, 1])
    plt.gca().invert_yaxis()
    plt.savefig(a_img)
    plt.close()

    return jsonify({
        "success": True,
        "human_plot": "static/trajectory_plots/Human_plot.png",
        "ai_plot": "static/trajectory_plots/AI_plot.png",
        "raw_human": h_img,
        "raw_ai": a_img,
        "human_score": hs,
        "ai_score": as_
    })


# ======================================================
# Start server
# ======================================================
if __name__ == "__main__":
    # When you run: python server.py
    # terminal will print the correct password for debugging
    app.run(host="0.0.0.0", port=9000, debug=True)