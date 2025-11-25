import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to support the Web service environment


# --------------------------
#  Calculate speed, acceleration and curvature
# --------------------------
def compute_metrics(points):

    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]

    dx = np.diff(x)
    dy = np.diff(y)

    speed = np.sqrt(dx**2 + dy**2)

    accel = np.diff(speed)

    epsilon = 1e-6
    curvature = np.abs(
        (dx[:-1] * np.diff(dy) - dy[:-1] * np.diff(dx)) /
        (speed[:-1]**3 + epsilon)
    )

    if len(speed) > 5:
        speed = savgol_filter(speed, 5, 2)
    if len(accel) > 5:
        accel = savgol_filter(accel, 5, 2)
    if len(curvature) > 5:
        curvature = savgol_filter(curvature, 5, 2)

    return speed, accel, curvature


# --------------------------
#  AI / HUMAN Classification Logic (New Version)
# --------------------------
def score_ai(speed, accel, curvature):
    """
    AI judgment logic redesigned according to real image characteristics

    The higher the return score, the more like AI (0~1)
    """

    ai_score = 0.0

    # -------------- AI strong characteristics (according to your figure）----------------

    # 1）The speed peak of AI is very huge (>80)
    if np.max(speed) > 80:
        ai_score += 0.30

    # 2）The acceleration peak of AI is also particularly large (>40)
    if np.max(np.abs(accel)) > 40:
        ai_score += 0.25

    # 3）The curvature peak of AI usually exceeds 1.0 (angle characteristic)
    if np.max(curvature) > 1.0:
        ai_score += 0.25

    # 4）The static ratio of AI is high (humans will not stay still for a long time)
    speed_zero_ratio = np.mean(speed < 0.1)
    if speed_zero_ratio > 0.25:
        ai_score += 0.10

    # 5）AI curvature oscillation is very regular (std is small but there are many peaks)
    curv_std = np.std(curvature)
    curv_peak_count = np.sum(curvature > 0.5)

    if curv_peak_count > 10 and curv_std < 0.3:
        ai_score += 0.10


    # --------------Human strong characteristics (deduct points）----------------

    # The peak of human speed is generally small (<25)
    if np.max(speed) < 25:
        ai_score -= 0.15

    # Human acceleration is relatively small (usually <15)
    if np.max(np.abs(accel)) < 15:
        ai_score -= 0.15

    # The human curvature peak is generally low (<0.7)
    if np.max(curvature) < 0.7:
        ai_score -= 0.15

    # Large curvature jitter = obvious hand shaking
    curv_jitter = np.mean(np.abs(np.diff(curvature)))
    if curv_jitter > 0.05:
        ai_score -= 0.15


    # ------ Limited to 0~1 ------
    ai_score = np.clip(ai_score, 0, 1)

    return ai_score


# --------------------------
# Drawing function
# --------------------------
def plot_trajectory(name, speed, accel, curvature):

    plt.figure(figsize=(10, 10))

    plt.subplot(3,1,1)
    plt.plot(speed)
    plt.title(f"{name} Speed (Smoothed)")

    plt.subplot(3,1,2)
    plt.plot(accel)
    plt.title(f"{name} Acceleration (Smoothed)")

    plt.subplot(3,1,3)
    plt.plot(curvature)
    plt.title(f"{name} Curvature (Smoothed)")

    os.makedirs("static/trajectory_plots", exist_ok=True)
    filename = f"static/trajectory_plots/{name}_plot.png"
    plt.savefig(filename)
    print(f"The image has been saved. → {filename}")

    plt.close()


# --------------------------
# Main process
# --------------------------
if __name__ == "__main__":

    A_file = input("Enter the path of file A：")
    B_file = input("Enter the path of file B：")

    with open(A_file, "r") as f:
        A_points = json.load(f)["points"]

    with open(B_file, "r") as f:
        B_points = json.load(f)["points"]

    # Calculate the index
    A_speed, A_accel, A_curv = compute_metrics(A_points)
    B_speed, B_accel, B_curv = compute_metrics(B_points)

    # Classification probability
    A_ai = score_ai(A_speed, A_accel, A_curv)
    B_ai = score_ai(B_speed, B_accel, B_curv)
    plot_trajectory("A", A_speed, A_accel, A_curv)
    plot_trajectory("B", B_speed, B_accel, B_curv)
    print("\n==== AI Judging result ====\n")
    print(f"A ({A_file}) AI probability：{A_ai * 100:.1f}%")
    print(f"B ({B_file}) AI probability：{B_ai * 100:.1f}%")

    if A_ai > B_ai:
        print("\nFinal judgment: A is the AI trajectory, B is Human\n")
    else:
        print("\nFinal judgment: B is the AI trajectory and A is Human\n")
