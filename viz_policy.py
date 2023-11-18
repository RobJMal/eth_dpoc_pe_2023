import numpy as np
import matplotlib.pyplot as plt
import itertools
import matplotlib.patches as patches
import matplotlib
from matplotlib.widgets import Button
import matplotlib.image
import sys

import warnings

warnings.filterwarnings("ignore")


class VizPolicy:
    input_space = np.array([-1, 0, 1])
    colormap = matplotlib.colormaps["RdYlGn"].reversed()
    COLORMAP_SHIFT = 0
    fig, axs = [], []
    axbig = None
    idx_big = None

    def __init__(self, filename="workspace_.npz") -> None:
        self.file = np.load(filename)
        J = self.file["J"]
        u = self.file["u"]
        self.cities = self.file["LOC_CITIES"]
        self.time_period = self.file["T"]

        self.mapWorld = np.zeros(
            (self.file["D"], self.file["N"], self.file["M"]), dtype=int
        )

        t = np.arange(0, self.file["T"])
        z = np.arange(0, self.file["D"])
        y = np.arange(0, self.file["N"])
        x = np.arange(0, self.file["M"])
        self.stateSpace = np.array(list(itertools.product(t, z, y, x)))

        self.current_time = 0

        self.map_J = []
        self.map_u = []
        for idx_t in range(self.time_period):
            self.map_J.append(
                self.match_to_map(J, self.stateSpace, self.mapWorld, idx_t)
            )
            self.map_u.append(
                self.match_to_map(u, self.stateSpace, self.mapWorld, idx_t)
            )

        self.minJ = np.array(self.map_J).min()
        self.maxJ = np.array(self.map_J).max()

    def increment_time(self):
        self.current_time += 1
        self.current_time = (self.current_time) % self.time_period

    def decrement_time(self):
        self.current_time -= 1
        self.current_time = (self.current_time) % self.time_period

    def match_to_map(self, A, stateSpace, mapWorld, time):
        mapA = np.zeros_like(mapWorld, dtype=np.float32)
        for idx in range(len(stateSpace)):
            b = stateSpace[idx]
            if b[0] != time:
                continue
            mapA[b[1], b[2], b[3]] = A[idx]
        return mapA

    def onClick_subplot(self, event):
        for i, ax in enumerate(self.axs[:-1]):
            if event.inaxes == ax[0]:
                self.idx_big = self.mapWorld.shape[0] - i - 1
                self.draw_level(self.axbig, self.mapWorld.shape[0] - i - 1)
        plt.show()

    def draw_level(self, ax, idx_d):
        ax.clear()
        ax.set_aspect("equal")
        if ax == self.axbig:
            ax.set_title("Time = " + str(self.current_time), fontweight="bold")

        for idx_y in range(self.mapWorld.shape[1]):
            for idx_x in range(self.mapWorld.shape[2]):
                cell_center_x = idx_x
                cell_center_y = idx_y

                normJ = (
                    self.map_J[self.current_time][idx_d, idx_y, idx_x] - self.minJ
                ) / (self.maxJ - self.minJ)
                color = self.colormap(
                    (normJ + self.COLORMAP_SHIFT) / (1 + self.COLORMAP_SHIFT)
                )

                cell = patches.Rectangle(
                    (idx_x - 0.5, idx_y - 0.5),
                    1,
                    1,
                    linewidth=1,
                    edgecolor="k",
                    facecolor=color,
                    zorder=0,
                )
                ax.add_patch(cell)
                if ax == self.axbig:
                    ax.annotate(
                        "{0:.1f}".format(
                            self.map_J[self.current_time][idx_d, idx_y, idx_x]
                        ),
                        (idx_x + 0.45, idx_y - 0.4),
                        fontsize=6,
                        color="k",
                        ha="right",
                    )

                arrow_length = 0.4
                a = self.input_space[
                    np.int16(self.map_u[self.current_time][idx_d, idx_y, idx_x])
                ]
                if a == 0:
                    circle = plt.Circle((cell_center_x, cell_center_y), 0.07, color="k")
                    ax.add_patch(circle)
                else:
                    ax.arrow(
                        cell_center_x,
                        cell_center_y - arrow_length * a * 0.8,
                        0,
                        arrow_length * a,
                        head_width=0.2,
                        head_length=0.2,
                        fc="k",
                        ec="k",
                        zorder=3,
                    )
                ax.use_sticky_edges = False
                ax.tick_params(
                    left=False,
                    right=False,
                    labelleft=False,
                    labelright=False,
                    labelbottom=False,
                    bottom=False,
                    top=False,
                    labeltop=False,
                )
                ax.set_xlim((-0.75, self.file["M"] - 0.25))
                ax.set_ylim((-0.75, self.file["N"] + 1.25))

                # ax.axis('off')

        if ax == self.axs[0, 0]:
            ax.set_title("Position of the Sun")
            for idx_x in range(self.mapWorld.shape[2]):
                cell_center_x = idx_x
                cell_center_y = self.file["N"] + 0.5

                cell = patches.Rectangle(
                    (cell_center_x - 0.5, cell_center_y - 0.5),
                    1,
                    1,
                    linewidth=1,
                    edgecolor="k",
                    facecolor="w",
                    zorder=0,
                )
                ax.add_patch(cell)

            sun_x = round(
                (self.file["M"] - 1)
                * ((self.time_period - 1) - self.current_time)
                / (self.time_period - 1)
            )
            sun_y = self.file["N"] + 0.5
            cell = patches.Rectangle(
                (sun_x - 0.5, sun_y - 0.5),
                1,
                1,
                linewidth=1,
                edgecolor="k",
                facecolor="#FFEA00",
                zorder=0,
            )
            ax.add_patch(cell)

        for city in self.cities:
            idx_x = city[1]
            idx_y = city[0]
            color = "b"
            cell = patches.Rectangle(
                (idx_x - 0.5, idx_y - 0.5),
                1,
                1,
                linewidth=2,
                edgecolor="b",
                facecolor=color,
                zorder=0,
                alpha=0.1 + 0.6 * (1 - idx_d / self.mapWorld.shape[0]),
            )
            ax.add_patch(cell)
        ax.set_ylabel("Level = " + str(idx_d), fontweight="bold")
        for spine in ax.spines.values():
            spine.set_edgecolor("w")

    def next_button_click(self, event):
        self.increment_time()
        self.redraw()

    def prev_button_click(self, event):
        self.decrement_time()
        self.redraw()

    def start_button_click(self, event):
        self.current_time = 0
        self.redraw()

    def end_button_click(self, event):
        self.current_time = self.time_period - 1
        self.redraw()

    def redraw(self):
        self.axbig.set_title("Time = " + str(self.current_time), fontweight="bold")
        for idx_d in range(self.mapWorld.shape[0]):
            idx_plot = self.mapWorld.shape[0] - idx_d - 1
            self.draw_level(self.axs[idx_plot, 0], idx_d)
        if self.idx_big is not None:
            self.draw_level(self.axbig, self.idx_big)
        plt.draw()

    def draw_policy(self):
        self.fig, self.axs = plt.subplots(self.mapWorld.shape[0] + 1, 2)

        # Draw levels
        for idx_d in range(self.mapWorld.shape[0]):
            idx_plot = self.mapWorld.shape[0] - idx_d - 1
            self.draw_level(self.axs[idx_plot, 0], idx_d)

        # Make space for the big axes
        gs = self.axs[1, 1].get_gridspec()
        # remove the underlying axes
        for ax in self.axs[:, 1]:
            ax.remove()
        self.axbig = self.fig.add_subplot(gs[:, 1])
        self.axbig.set_title("Time = " + str(self.current_time), fontweight="bold")
        self.axbig.annotate(
            "Click a plot on the left to enlarge\nThe controls below change the time step",
            [0.5, 0.5],
            va="center",
            ha="center",
        )
        self.axbig.tick_params(
            left=False,
            right=False,
            labelleft=False,
            labelright=False,
            labelbottom=False,
            bottom=False,
            top=False,
            labeltop=False,
        )
        self.axbig.axis("off")

        # Buttons
        self.ax_next_button = self.fig.add_axes([0.72, 0.06, 0.05, 0.05])
        self.next_button = Button(
            ax=self.ax_next_button,
            label=">",
            color="aliceblue",
            hovercolor="deepskyblue",
        )
        self.next_button.on_clicked(self.next_button_click)
        self.ax_prev_button = self.fig.add_axes([0.66, 0.06, 0.05, 0.05])
        self.prev_button = Button(
            ax=self.ax_prev_button,
            label="<",
            color="aliceblue",
            hovercolor="deepskyblue",
        )
        self.prev_button.on_clicked(self.prev_button_click)
        self.ax_start_button = self.fig.add_axes([0.60, 0.06, 0.05, 0.05])
        self.start_button = Button(
            ax=self.ax_start_button,
            label="|<",
            color="aliceblue",
            hovercolor="deepskyblue",
        )
        self.start_button.on_clicked(self.start_button_click)
        self.ax_end_button = self.fig.add_axes([0.78, 0.06, 0.05, 0.05])
        self.end_button = Button(
            ax=self.ax_end_button,
            label=">|",
            color="aliceblue",
            hovercolor="deepskyblue",
        )
        self.end_button.on_clicked(self.end_button_click)

        # Legend
        self.axs[-1, 0].imshow(matplotlib.image.imread("./legend.png"))
        self.axs[-1, 0].axis("off")

        self.fig.canvas.mpl_connect("button_press_event", self.onClick_subplot)
        self.fig.tight_layout()
        fig = plt.gcf()
        plt.show()


if __name__ == "__main__":
    filename = "./workspaces/workspace_.npz"
    if len(sys.argv) > 1:
        filename = sys.argv[1]

    vizPolicy = VizPolicy(filename)
    vizPolicy.draw_policy()
