import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from mesostat.visualization.mpl_colorbar import imshow_add_fake_color_bar
from mesostat.visualization.mpl_colors import sample_cmap


class BrainMap:
    def __init__(self):
        thisDir = os.path.abspath(os.getcwd())
        rootDir = os.path.dirname(os.path.dirname(thisDir))
        dataDir = os.path.join(rootDir, 'data_generic')

        # Get path to brain image
        self.imgpath = os.path.join(dataDir, 'mousebrain_coronal_allen.png')

        # Load brain coordinates
        self.brainCoordsPath = os.path.join(dataDir, 'mousebrain_coords.json')

        with open(self.brainCoordsPath, 'r') as f:
            self.brainCoords = json.load(f)

        # Dataholders
        self.areas = {}
        self.edges = {}
        self.triangles = {}

        # Constants
        self.areaRad = 50.0

    def _make_arrow(self, pos, col, str, ax):
        kw = dict(arrowstyle="Simple,tail_width=1,head_width=8,head_length=8", color=col)

        dist = np.linalg.norm(pos[1] - pos[0])
        eps = self.areaRad / dist
        posPrim = [(1 - eps) * pos[0] + eps * pos[1], eps * pos[0] + (1 - eps) * pos[1]]
        arrow123 = patches.FancyArrowPatch(posPrim[0], posPrim[1], **kw)  # , connectionstyle="arc3,rad=.5
        ax.add_patch(arrow123)

    def add_areas(self, areas, strengths):
        for area, str in zip(areas, strengths):
            assert area in self.brainCoords.keys()
            self.areas[area] = str

    def add_edges(self, edges, strengths):
        for (area1, area2), str in zip(edges, strengths):
            assert area1 in self.areas.keys()
            assert area2 in self.areas.keys()
            self.edges[(area1, area2)] = str

    def add_triangles(self, triangles, strengths):
        for (area1, area2, area3), str in zip(triangles, strengths):
            assert area1 in self.areas.keys()
            assert area2 in self.areas.keys()
            assert area3 in self.areas.keys()
            self.triangles[(area1, area2, area3)] = str

    def plot(self, cmap="jet", vmin=0, vmax=1):
        fig, ax = plt.subplots()

        # Plot background image
        image = plt.imread(self.imgpath)
        ax.imshow(image, origin='upper')
        ax.axis('off')

        # Add fake color bar for strength
        imshow_add_fake_color_bar(fig, ax, cmap=cmap, vmin=vmin, vmax=vmax)

        # Plot triangles (plot them first, as triangles should be below the rest of the stuff)
        for (area1, area2, area3), str in self.triangles.items():
            p1 = np.array(self.brainCoords[area1])
            p2 = np.array(self.brainCoords[area2])
            p3 = np.array(self.brainCoords[area3])
            color = sample_cmap(cmap, [str], vmin=vmin, vmax=vmax)[0]

            t1 = plt.Polygon([p1, p2, p3], color=color, alpha=0.3, linewidth=0)
            ax.add_patch(t1)

        # Plot nodes
        for area, str in self.areas.items():
            color = sample_cmap(cmap, [str], vmin=vmin, vmax=vmax)[0]
            circ = plt.Circle(self.brainCoords[area], self.areaRad, color=color, alpha=0.5)
            ax.add_patch(circ)

        # Plot edges
        for (area1, area2), str in self.edges.items():
            p1 = np.array(self.brainCoords[area1])
            p2 = np.array(self.brainCoords[area2])
            color = sample_cmap(cmap, [str], vmin=vmin, vmax=vmax)[0]
            self._make_arrow([p1, p2], color, str, ax)

        return fig, ax


bm = BrainMap()
bm.add_areas(['S1', 'M1', 'PrL'], [1.0, 1.0, 1.0])
bm.add_edges([['S1', 'PrL'], ['M1', 'PrL']], [0.002, 0.2])
bm.add_triangles([['S1', 'M1', 'PrL']], [0.15])

fig, ax = bm.plot()
plt.savefig('yolo.png', dpi=300)
plt.show()
