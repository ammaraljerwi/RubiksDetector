class CONSTANTS:
    face_size = 400

    cube_classNames = ["Rubik's Cube"]
    sticker_classNames = ["green", "blue", "orange", "red", "yellow", "white"]
    idx_to_color = {0: "G", 1: "B", 2: "O", 3: "R", 4: "Y", 5: "W"}
    colors = {
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "orange": (0, 165, 255),
        "red": (0, 0, 255),
        "yellow": (0, 255, 255),
        "white": (255, 255, 255),
    }
    rgb_to_color = {v: k for k, v in colors.items()}
    color_to_idx = {k: i for i, k in enumerate(colors.keys())}
    num_to_color = {(i + 1): k for i, k in enumerate(colors.keys())}
