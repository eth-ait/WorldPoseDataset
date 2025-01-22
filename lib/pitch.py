import numpy as np

HALF_LINE_WIDTH = 0.05
X, Y = (52.50 - HALF_LINE_WIDTH, 34.00 - HALF_LINE_WIDTH)
PEN_X, PEN_Y = (X - 16.5, 20.16)
GL_X, GL_Y = (X - 5.5, 9.16)
PEN_SPOT = X - 11
R = 9.15

SOCCER_PITCH_EDGES = np.array([
    # Home
    (0, 2), (2, 6), (6, 9), (9, 5), (5, 1), (2, 3),
    (3, 4), (4, 5), (6, 7), (7, 8), (8, 9),
    # Away
    (10, 12), (12, 16), (16, 19), (19, 15), (15, 11), (12, 13),
    (13, 14), (14, 15), (16, 17), (17, 18), (18, 19),
    # middle
    (0, 20), (1, 24), (10, 20), (11, 24), 
    (20, 21), (21, 22), (22, 23), (23, 24),
])

SOCCER_PITCH_KEYPOINTS = np.array(
    [
        # Home
        ## Outside Lines (0 - 1)
        [X, -Y, 0], [X, Y, 0],
        ## Penalty Zone (2 - 5)
        [X, -PEN_Y, 0], [PEN_X, -PEN_Y, 0], [PEN_X, PEN_Y, 0], [X, PEN_Y, 0],
        ## Goal Zone (6 - 9)
        [X, -GL_Y, 0], [GL_X, -GL_Y, 0], [GL_X, GL_Y, 0], [X, GL_Y, 0],
        # Away
        ## Outside Lines (10 - 11)
        [-X, -Y, 0], [-X, Y, 0],
        ## Penalty Zone (12 - 15)
        [-X, -PEN_Y, 0], [-PEN_X, -PEN_Y, 0], [-PEN_X, PEN_Y, 0], [-X, PEN_Y, 0],
        ## Goal Zone (16 - 19)
        [-X, -GL_Y, 0], [-GL_X, -GL_Y, 0], [-GL_X, GL_Y, 0], [-X, GL_Y, 0],
        # Other
        ## Half-way Line (20 - 24)
        [0, -Y, 0], [0, -R, 0], [0, 0, 0], [0, R, 0], [0, Y, 0],
        ## Penalty Spot (Home) (25)
        [PEN_SPOT, 0, 0],
        ## Penalty Spot (Away) (26)
        [-PEN_SPOT, 0, 0],
    ]
)

class SoccerPitch:
    def __init__(self) -> None:
        self.pts = SOCCER_PITCH_KEYPOINTS.copy()
        self.edges = SOCCER_PITCH_EDGES.copy()
        self.lines = self.create_lines(self.pts)

    def create_lines(self, points_3d):
        """Create lines for visualization."""

        # add lines
        lines = []
        for i, j in self.edges:
            p1 = points_3d[i]
            p2 = points_3d[j]
            d = int(np.linalg.norm(p1 - p2) + 0.5)
            lines.append(np.linspace(p1, p2, num=d))

        # add arcs
        ARC = np.linspace(0.70 * np.pi, 1.30 * np.pi, 18)
        R = 9.15
        c = points_3d[25]
        lines.append(
            c[None, :] + R * np.stack([np.cos(ARC), np.sin(ARC), np.zeros_like(ARC)]).T
        )

        c = points_3d[26]
        lines.append(
            c[None, :] + R * np.stack([-np.cos(ARC), np.sin(ARC), np.zeros_like(ARC)]).T
        )

        # add circles
        CIRCLE = np.linspace(0, 2 * np.pi, 58)
        c = points_3d[22]
        lines.append(
            c[None, :]
            + R * np.stack([np.cos(CIRCLE), np.sin(CIRCLE), np.zeros_like(CIRCLE)]).T
        )
        return lines
