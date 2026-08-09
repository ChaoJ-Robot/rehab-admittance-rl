"""Seeded, physically walkable maze layouts for planar rehabilitation tasks."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class MazeLayout:
    """Continuous wall segments and an optimal cell-centre route."""

    walls: list[list[float]]
    waypoints: list[list[float]]
    start: list[float]
    goal: list[float]
    seed: int


def generate_grid_maze(
    *,
    columns: int,
    rows: int,
    seed: int,
    min_x: float = 0.15,
    max_x: float = 0.60,
    min_y: float = -0.22,
    max_y: float = 0.22,
    home: tuple[float, float] = (0.35, 0.0),
) -> MazeLayout:
    """Generate a perfect maze and the shortest route from home to a far goal.

    A depth-first spanning tree creates the passages; a breadth-first pass then
    selects a distant goal and reconstructs the optimal route. Returning the
    seed makes every prescribed maze reproducible and auditable.
    """

    if columns < 3 or rows < 3:
        raise ValueError("maze grid must be at least 3 x 3")
    rng = random.Random(int(seed))
    width = (max_x - min_x) / columns
    height = (max_y - min_y) / rows

    def center(cell: tuple[int, int]) -> tuple[float, float]:
        column, row = cell
        return min_x + (column + 0.5) * width, min_y + (row + 0.5) * height

    start = min(
        ((column, row) for column in range(columns) for row in range(rows)),
        key=lambda cell: (center(cell)[0] - home[0]) ** 2 + (center(cell)[1] - home[1]) ** 2,
    )
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    passages: set[frozenset[tuple[int, int]]] = set()
    visited = {start}
    stack = [start]
    while stack:
        current = stack[-1]
        candidates = []
        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < columns and 0 <= neighbor[1] < rows and neighbor not in visited:
                candidates.append(neighbor)
        if not candidates:
            stack.pop()
            continue
        neighbor = rng.choice(candidates)
        passages.add(frozenset((current, neighbor)))
        visited.add(neighbor)
        stack.append(neighbor)

    # The robot home may sit exactly on a grid boundary (for example y=0 in
    # an even-row grid). Open that boundary so the initial pose is never
    # rendered or scored as touching a wall.
    start_left = min_x + start[0] * width
    start_right = start_left + width
    start_bottom = min_y + start[1] * height
    start_top = start_bottom + height
    boundary_neighbors = []
    if abs(home[0] - start_left) < 1e-9 and start[0] > 0:
        boundary_neighbors.append((start[0] - 1, start[1]))
    if abs(home[0] - start_right) < 1e-9 and start[0] + 1 < columns:
        boundary_neighbors.append((start[0] + 1, start[1]))
    if abs(home[1] - start_bottom) < 1e-9 and start[1] > 0:
        boundary_neighbors.append((start[0], start[1] - 1))
    if abs(home[1] - start_top) < 1e-9 and start[1] + 1 < rows:
        boundary_neighbors.append((start[0], start[1] + 1))
    for neighbor in boundary_neighbors:
        passages.add(frozenset((start, neighbor)))

    graph: dict[tuple[int, int], list[tuple[int, int]]] = {
        (column, row): [] for column in range(columns) for row in range(rows)
    }
    for passage in passages:
        first, second = tuple(passage)
        graph[first].append(second)
        graph[second].append(first)
    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    distances = {start: 0}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for neighbor in graph[current]:
            if neighbor in parents:
                continue
            parents[neighbor] = current
            distances[neighbor] = distances[current] + 1
            queue.append(neighbor)
    goal_cell = max(distances, key=lambda cell: distances[cell])
    route = []
    cursor: tuple[int, int] | None = goal_cell
    while cursor is not None:
        route.append(cursor)
        cursor = parents[cursor]
    route.reverse()

    walls: list[list[float]] = []
    for column in range(columns):
        for row in range(rows):
            cell = (column, row)
            left = min_x + column * width
            right = left + width
            bottom = min_y + row * height
            top = bottom + height
            if column == 0:
                walls.append([left, bottom, left, top])
            if row == 0:
                walls.append([left, bottom, right, bottom])
            right_neighbor = (column + 1, row)
            if column == columns - 1 or frozenset((cell, right_neighbor)) not in passages:
                walls.append([right, bottom, right, top])
            top_neighbor = (column, row + 1)
            if row == rows - 1 or frozenset((cell, top_neighbor)) not in passages:
                walls.append([left, top, right, top])

    start_center = center(start)
    route_points = [[float(home[0]), float(home[1]), 0.0]]
    if abs(start_center[0] - home[0]) > 1e-9 or abs(start_center[1] - home[1]) > 1e-9:
        route_points.append([start_center[0], start_center[1], 0.0])
    route_points.extend([[*center(cell), 0.0] for cell in route[1:]])
    return MazeLayout(
        walls=walls,
        waypoints=route_points,
        start=route_points[0],
        goal=route_points[-1],
        seed=int(seed),
    )
