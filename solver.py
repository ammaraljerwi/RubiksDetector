import magiccube

from constants import CONSTANTS


def convert_state_to_cube(state):
    state_str = ""
    for i in range(len(state)):
        for j in range(len(state[i])):
            state_str += CONSTANTS.idx_to_color[int(state[i][j].item())]
    cube = magiccube.Cube(3, state_str)
    return cube


def solve_cube(state):
    """solve cube"""
    cube = convert_state_to_cube(state)
    solver = magiccube.BasicSolver(cube)
    solution = solver.solve()
    return solution
