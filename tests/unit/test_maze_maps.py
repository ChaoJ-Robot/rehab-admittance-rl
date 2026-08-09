"""Every maze map must have a physically walkable reference route.

Regression guard for the class of bug where a waypoint segment crosses a
wall: for each map in configs/tasks.yaml the polyline
(robot start -> waypoint 1 -> ... -> waypoint N) must not strictly intersect
any wall segment, and every waypoint must sit inside the outer boundary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rehab_sim.config import load_yaml
from rehab_sim.tasks.maze import generate_grid_maze

_ROOT = Path(__file__).resolve().parents[2]
_START = np.array([0.35, 0.0], dtype=np.float64)
_EPS = 1e-9


def _strictly_intersect(p1, p2, p3, p4) -> bool:
    """True when segments cross through each other (endpoint touches exempt)."""

    def cross(origin, a, b) -> float:
        return float(
            (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])
        )

    d1 = cross(p1, p2, p3)
    d2 = cross(p1, p2, p4)
    d3 = cross(p3, p4, p1)
    d4 = cross(p3, p4, p2)
    return ((d1 > _EPS and d2 < -_EPS) or (d1 < -_EPS and d2 > _EPS)) and (
        (d3 > _EPS and d4 < -_EPS) or (d3 < -_EPS and d4 > _EPS)
    )


def _maps() -> dict[str, dict]:
    config = load_yaml(_ROOT / "configs" / "tasks.yaml")
    return config["tasks"]["maze_navigation"]["maps"]


def test_every_maze_has_at_least_five_waypoints() -> None:
    for name, layout in _maps().items():
        assert len(layout["waypoints"]) >= 5, f"{name} is too simple"


def test_every_maze_route_is_walkable() -> None:
    for name, layout in _maps().items():
        walls = [
            (
                np.asarray(wall[:2], dtype=np.float64),
                np.asarray(wall[2:], dtype=np.float64),
            )
            for wall in layout["walls"]
        ]
        waypoints = [np.asarray(point[:2], dtype=np.float64) for point in layout["waypoints"]]
        segments = [(_START, waypoints[0]), *zip(waypoints, waypoints[1:], strict=False)]
        for index, (begin, end) in enumerate(segments):
            for wall_a, wall_b in walls:
                assert not _strictly_intersect(begin, end, wall_a, wall_b), (
                    f"map {name}: segment {index} {begin.tolist()} -> {end.tolist()} "
                    f"crosses wall {wall_a.tolist()} -> {wall_b.tolist()}"
                )


def test_every_maze_waypoint_inside_bounds() -> None:
    for name, layout in _maps().items():
        for point in layout["waypoints"]:
            x, y = float(point[0]), float(point[1])
            assert 0.15 < x < 0.60 and -0.22 < y < 0.22, (
                f"map {name}: waypoint {point} outside the outer walls"
            )


def test_every_maze_wall_inside_bounds() -> None:
    for name, layout in _maps().items():
        for wall in layout["walls"]:
            for coordinate in (wall[:2], wall[2:]):
                x, y = float(coordinate[0]), float(coordinate[1])
                assert 0.15 - 1e-9 <= x <= 0.60 + 1e-9 and -0.22 - 1e-9 <= y <= 0.22 + 1e-9, (
                    f"map {name}: wall endpoint {coordinate} outside the workspace"
                )


def test_procedural_maze_is_reproducible_and_walkable() -> None:
    first = generate_grid_maze(columns=8, rows=6, seed=17)
    repeated = generate_grid_maze(columns=8, rows=6, seed=17)
    different = generate_grid_maze(columns=8, rows=6, seed=18)
    assert first == repeated
    assert first.walls != different.walls
    assert len(first.walls) >= 50
    assert len(first.waypoints) >= 10
    walls = [
        (np.asarray(wall[:2], dtype=np.float64), np.asarray(wall[2:], dtype=np.float64))
        for wall in first.walls
    ]
    points = [np.asarray(point[:2], dtype=np.float64) for point in first.waypoints]
    for begin, end in zip(points, points[1:], strict=False):
        assert not any(_strictly_intersect(begin, end, wall_a, wall_b) for wall_a, wall_b in walls)
