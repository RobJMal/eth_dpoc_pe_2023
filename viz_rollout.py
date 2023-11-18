import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import tqdm
import os
import datetime
import sys


class VizRollout:
    input_space = np.array([-1, 0, 1])
    colormap = matplotlib.colormaps["RdYlGn"].reversed()
    COLORMAP_SHIFT = 0

    def __init__(self, filename) -> None:
        self.file = np.load(filename)
        self.current_time = 0
        self.frame_counter = 0
        self.P = self.file["P"]
        self.J = self.file["J"]
        self.u = self.file["u"].astype(int)
        self.D = self.file["D"]
        self.time_period = self.file["T"]
        self.cities = self.file["LOC_CITIES"]
        self.mapWorld = np.zeros(
            (self.file["D"], self.file["N"], self.file["M"]), dtype=int
        )
        self.balloon = [
            np.random.randint(0, self.file["D"]),
            np.random.randint(0, self.file["N"]),
            np.random.randint(0, self.file["M"]),
        ]

        t = np.arange(0, self.file["T"])
        z = np.arange(0, self.file["D"])
        y = np.arange(0, self.file["N"])
        x = np.arange(0, self.file["M"])
        self.stateSpace = np.array(list(itertools.product(t, z, y, x)))

        self.map_J = []
        self.map_u = []
        for idx_t in range(self.file["T"]):
            self.map_J.append(
                self.match_to_map(self.J, self.stateSpace, self.mapWorld, idx_t)
            )
            self.map_u.append(
                self.match_to_map(self.u, self.stateSpace, self.mapWorld, idx_t)
            )

        self.map_J = np.array(self.map_J)
        self.minJ = self.map_J.min()
        self.maxJ = self.map_J.max()

        plt.ion()
        self.fig, self.axs = plt.subplots(self.mapWorld.shape[0])

    def check_admissible_policy(self):
        for state, input in enumerate(self.u):
            if np.sum(self.P[state, :, input]) == 0:
                return False
        return True


    def increment_time(self):
        self.current_time += 1
        self.current_time = (self.current_time) % self.time_period

    def match_to_map(self, A, stateSpace, mapWorld, time):
        mapA = np.zeros_like(mapWorld, dtype=np.float32)
        for idx in range(len(stateSpace)):
            b = stateSpace[idx]
            if b[0] != time:
                continue
            mapA[b[1], b[2], b[3]] = A[idx]
        return mapA

    # Two helper functions that relate 1d-indices to states
    def idx2state(self, idx):
        """The index corresponding to [z, y, x, t].
        Assuming ordering from last dim changing fastest to first dim changing slowest
        """
        state = np.empty(4)
        for i, j in enumerate(
            [self.file["M"], self.file["N"], self.file["D"], self.file["T"]]
        ):
            state[3 - i] = idx % j
            idx = idx // j
        return state

    def state2idx(self, state):
        idx = 0
        factor = 1
        for i, j in enumerate(
            [self.file["M"], self.file["N"], self.file["D"], self.file["T"]]
        ):
            idx += state[3 - i] * factor
            factor *= j
        return idx

    def evolve_system(self):
        print("Generating frames...")
        for i in tqdm.tqdm(range(25)):
            idx_state = self.state2idx(
                [self.current_time, self.balloon[0], self.balloon[1], self.balloon[2]]
            )
            idx_next = np.nonzero(self.P[idx_state, :, self.u[idx_state]])
            p_next = self.P[idx_state, idx_next, self.u[idx_state]][0]
            idx_next = idx_next[0][np.random.choice(range(len(p_next)), 1, p=p_next)[0]]

            a = self.input_space[
                np.int16(
                    self.map_u[self.current_time][
                        self.balloon[0], self.balloon[1], self.balloon[2]
                    ]
                )
            ]
            # print(a, self.stateSpace[idx_state], self.stateSpace[idx_next])
            self.balloon = [
                self.stateSpace[idx_next][1],
                self.stateSpace[idx_next][2],
                self.stateSpace[idx_next][3],
            ]
            self.draw_frame()
            plt.savefig("./imgs/" + str(self.frame_counter).zfill(2))
            self.frame_counter += 1

            self.increment_time()

    def draw_frame(self):
        for idx_d in range(self.D):
            idx_plot = self.mapWorld.shape[0] - idx_d - 1
            self.draw_level(self.axs[idx_plot], idx_d)
        balloon = patches.Circle(
            (self.balloon[2], self.balloon[1]),
            radius=0.35,
            facecolor=[1, 0, 0],
            edgecolor="k",
            zorder=4,
            linewidth=2,
        )
        self.axs[self.mapWorld.shape[0] - 1 - self.balloon[0]].add_patch(balloon)

        self.fig.suptitle("Time " + str(self.current_time))
        plt.show(block=False)

    def draw_level(self, ax, idx_d):
        ax.clear()
        ax.set_aspect("equal")

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

        if ax == self.axs[0]:
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

    def make_video(self):
        image_folder = "./imgs"
        tim = str(datetime.datetime.now().time().strftime("%H_%M_%S"))
        video_name = "./videos/vid_" + tim + ".avi"

        images = sorted(
            [img for img in os.listdir(image_folder) if img.endswith(".png")]
        )
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, 0, 1, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()


if __name__ == "__main__":
    filename = "./workspaces/workspace_.npz"
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    vr = VizRollout(filename)
    if vr.check_admissible_policy():
        vr.draw_frame()
        vr.evolve_system()
        vr.make_video()
    else:
        print('The policy rollout is not possible because the policy is not admissible with respect to the transition probabilities.')
        print('Revisit your solution, run main.py and then viz_rollout.py.')