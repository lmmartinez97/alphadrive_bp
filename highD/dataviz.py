import numpy as np
import pandas as pd
import matplotlib as mlp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.patches import Rectangle, Ellipse
from pprint import pprint
from rich import print
from time import time

from read_csv import read_track_csv, read_static_info, read_meta_info

sns.set()
sns.set_style("whitegrid")
sns.set_context("paper")
sns.color_palette("hls", 8)


def print_bl():
    print("\n")


class SceneData:
    def __init__(self, df, static_info, video_info):
        self.df = df
        self.static_info = static_info
        self.video_info = video_info
        self.frame_delay = 1000 / self.video_info["frameRate"]
        self.df_list = self.get_df_list()
        self.total_frame_number = len(self.df_list)
        self.df_direction_list = self.get_df_direction_list()
        self.bg_image = None

    def get_df_list(self):
        self.df_list = [
            None
        ] * self.df.frame.nunique()  # Preallocate list to improve speed - number of dataframes is equal to that of video frames
        for frame_number in range(
            1, self.df.frame.nunique() + 1, 1
        ):  # Iterate through every video frame in the dataframe
            self.df_list[frame_number - 1] = self.df[
                np.in1d(self.df["frame"].values, [frame_number])
            ]  # Select rows of dataframe with df["frame"] == frame_number
        return self.df_list

    def get_df_direction_list(self):
        eastbound_vehicles = []
        westbound_vehicles = []
        for vehicle_id in self.static_info.keys():
            if self.static_info[vehicle_id]["drivingDirection"] == 2.0:
                eastbound_vehicles.append(vehicle_id)
            else:
                westbound_vehicles.append(vehicle_id)

        self.df_direction_list = [None] * df.frame.nunique()
        dummy_east, dummy_west = None, None
        for idx in range(len(self.df_list)):
            dummy_east = self.df_list[idx][
                [ident in eastbound_vehicles for ident in self.df_list[idx].id]
            ]  # eastbound vehicles
            dummy_west = self.df_list[idx][
                [ident in westbound_vehicles for ident in self.df_list[idx].id]
            ]  # westbound vehicles
            self.df_direction_list[idx] = [dummy_east, dummy_west]
        return self.df_direction_list

    def get_background_img(self, path):
        self.bg_image = plt.imread(path)

    def plot_frame(self, frame_number=None):
        if frame_number is None:
            frame_number = np.random.randint(0, self.total_frame_number)
        eastbound_df, westbound_df = self.df_direction_list[frame_number]
        fig, ax = plt.subplots()
        if self.bg_image is not None:
            ax.imshow(self.bg_image)
        for index, vehicle in eastbound_df.iterrows():
            rect = Rectangle(
                (int(vehicle.x * 2.45), int(vehicle.y * 2.45)),
                int(vehicle.width * 2.45),
                int(vehicle.height * 2.45),
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)
        for index, vehicle in westbound_df.iterrows():
            rect = Rectangle(
                (int(vehicle.x * 2.45), int(vehicle.y * 2.45)),
                int(vehicle.width * 2.45),
                int(vehicle.height * 2.45),
                linewidth=1,
                edgecolor="g",
                facecolor="none",
            )
            ax.add_patch(rect)
        plt.show()

    def plot_frame_anim(self, frame_number, ax):
        ax.clear()
        ax.imshow(self.bg_image)
        eastbound_df, westbound_df = self.df_direction_list[frame_number]
        for index, vehicle in eastbound_df.iterrows():
            rect = Rectangle(
                (int(vehicle.x * 2.45), int(vehicle.y * 2.45)),
                int(vehicle.width * 2.45),
                int(vehicle.height * 2.45),
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)
        for index, vehicle in westbound_df.iterrows():
            rect = Rectangle(
                (int(vehicle.x * 2.45), int(vehicle.y * 2.45)),
                int(vehicle.width * 2.45),
                int(vehicle.height * 2.45),
                linewidth=1,
                edgecolor="g",
                facecolor="none",
            )
            ax.add_patch(rect)

    def create_gif(self):
        fig, ax = plt.subplots()
        fig.set_tight_layout(True)
        if self.bg_image is not None:
            ax.imshow(self.bg_image)
        anim = animation.FuncAnimation(
            fig=fig,
            func=self.plot_frame_anim,
            frames=self.total_frame_number,
            interval=self.frame_delay,
        )
        plt.show()


def show_gif(scene_data):
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(scene_data.bg_image)
    anim = animation.FuncAnimation(
        fig=fig,
        func=scene_data.plot_frame_anim,
        fargs=(ax,),
        frames=scene_data.total_frame_number,
        interval=scene_data.frame_delay / 1000,
        blit=False,
    )
    if 0:
        writer = animation.PillowWriter(
            fps=scene_data.video_info["frameRate"],
            metadata=dict(artist="Me"),
            bitrate=900,
        )
        anim.save("test.gif", writer=writer)
    plt.show()

dataset_location = "/home/lmmartinez/Tesis/datasets/highD/data_raw/"

start = time()
df = pd.read_csv(dataset_location + "01_tracks.csv")
tracks = read_track_csv(dataset_location + "01_tracks.csv")
static_info = read_static_info(dataset_location + "01_tracksMeta.csv")
video_info = read_meta_info(dataset_location + "01_recordingMeta.csv")
im_path = dataset_location + "01_highway.png"

scene_data = SceneData(df=df, static_info=static_info, video_info=video_info)
scene_data.get_background_img(im_path)
end = time()

print("Time elapsed is:", start - end)

# show_gif(scene_data=scene_data)
