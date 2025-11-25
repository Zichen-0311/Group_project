import numpy as np
import torch
import json
import pygame
from mouse_trajectory_model import TrajectoryGenerator, generate_trajectory

# ------------------ Nine palace grid coordinate mapping ------------------
def build_nine_grid(cell_size=120):
    OFFSET = cell_size // 2  # Click the cell center point

    grid = {
        "1": np.array([cell_size * 0 + OFFSET, cell_size * 0 + OFFSET]),
        "2": np.array([cell_size * 1 + OFFSET, cell_size * 0 + OFFSET]),
        "3": np.array([cell_size * 2 + OFFSET, cell_size * 0 + OFFSET]),
        "4": np.array([cell_size * 0 + OFFSET, cell_size * 1 + OFFSET]),
        "5": np.array([cell_size * 1 + OFFSET, cell_size * 1 + OFFSET]),
        "6": np.array([cell_size * 2 + OFFSET, cell_size * 1 + OFFSET]),
        "7": np.array([cell_size * 0 + OFFSET, cell_size * 2 + OFFSET]),
        "8": np.array([cell_size * 1 + OFFSET, cell_size * 2 + OFFSET]),
        "9": np.array([cell_size * 2 + OFFSET, cell_size * 2 + OFFSET]),
        "0": np.array([cell_size * 1 + OFFSET, cell_size * 3 + OFFSET]),
    }
    return grid


# ------------------ AI trajectory generation ------------------
def generate_sequence_trajectory(sequence, model_path="trajectory_model.pth", save_json=True):
    grid = build_nine_grid()

    # Model loading
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TrajectoryGenerator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"\nInput sequence: {sequence}")
    points = [grid[num] for num in sequence]

    full_traj = []

    for i in range(len(points)-1):
        start = points[i]
        end = points[i+1]
        
        ai_traj = generate_trajectory(model, start, end, device=device).astype(int)

        if i > 0:
            ai_traj = ai_traj[1:]
        full_traj.extend(ai_traj.tolist())

        print(f"Paragraph {i+1}: {sequence[i]} -> {sequence[i+1]} Bring into being {len(ai_traj)} ")

    if save_json:
        filename = f"ai_traj_{sequence}.json"
        with open(filename, 'w') as f:
            json.dump({"sequence": sequence, "points": full_traj}, f, indent=4)
        print(f"The AI track has been saved to: {filename}")

    return np.array(full_traj)


# ------------------User's real track collection ------------------
def record_human_trajectory(sequence, width=800, height=600, save_json=True):
    grid = build_nine_grid(width, height)

    pygame.init()
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 32)

    points = []
    seq_index = 0
    target = grid[sequence[seq_index]]

    print("\nPlease click in order.：", " → ".join(sequence))

    running = True
    tracking = False  # Whether the full trajectory is being recorded

    while running:
        mouse_pos = np.array(pygame.mouse.get_pos())

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # --------------------------
            # Click the target point (start or continue the process)
            # --------------------------
            if event.type == pygame.MOUSEBUTTONUP:

                # Click the correct point
                if np.linalg.norm(mouse_pos - target) < 40:
                    print(f"Clicked successfully: {sequence[seq_index]}")

                    #The first point: start recording the whole track
                    if seq_index == 0:
                        tracking = True
                        print("Start to record the trajectory...")

                    seq_index += 1

                    # If the last point has been completed → End the record
                    if seq_index == len(sequence):
                        tracking = False
                        running = False
                        print("The last point has been reached, and the track recording has been stopped.")
                        break

                    # The next target point
                    target = grid[sequence[seq_index]]

        # --------------------------
        # As long as tracking=True, each frame will record the movement of the mouse.
        # --------------------------
        if tracking:
            points.append(mouse_pos.tolist())

        # ----Map ----
        screen.fill((255,255,255))
        for key, pos in grid.items():
            if seq_index < len(sequence):
                color = (0,255,0) if key == sequence[seq_index] else (200,200,200)
            else:
                color = (200,200,200)
            pygame.draw.circle(screen, color, pos, 10)

        pygame.display.flip()
        clock.tick(240)

    pygame.quit()

    # Save data
    if save_json:
        filename = f"human_traj_{sequence}.json"
        with open(filename, "w") as f:
            json.dump({"sequence": sequence, "points": points}, f, indent=4)
        print(f"\nThe track has been saved to: {filename}")

    return np.array(points)



if __name__ == "__main__":
    user_input = input("Please enter the sequence of the nine palace grids (e.g. 2518)：\n> ").strip()

    if not user_input.isdigit() or any(d not in "123456789" for d in user_input) or len(user_input) < 2:
        print("Input error! Only numbers 1~9, at least two, are allowed.")
        exit()

    print("\nPlease select the mode:\n1. Generate AI track\n2. Users collect the real track")
    mode = input("Input 1 or 2:\n> ").strip()

    if mode == "1":
        traj = generate_sequence_trajectory(user_input)
    elif mode == "2":
        traj = record_human_trajectory(user_input)
    else:
        print("The input is invalid, exit the program.")
