import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib as mlp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import os
import seaborn as sns

from copy import copy
from IPython.display import HTML
from matplotlib.patches import Rectangle, Ellipse
from pprint import pprint
from rich import print
from sys import platform
from time import time

from read_csv import read_track_csv, read_static_info, read_meta_info

sns.set()
sns.set_style("whitegrid")
sns.set_context("paper")
sns.color_palette("hls", 8)

mlp.rcParams["animation.embed_limit"] = (2**128)  # Increased animation size to save gifs if necessary
pd.set_option('mode.chained_assignment', None)

class SceneData:
    '''
    A class that represents an entire recording of the highD dataset.

    Attributes:
        dataset_location: a path to the directory in which the dataset is stored.
        dataset_index: index of the recording that is to be addressed
        sampling_period: time period between sampled frames. Needs to be a multiple of 40 ms
    '''
    def __init__(self, dataset_location = None, dataset_index = None, sampling_period = 40):
        self.dataset_index  = dataset_index
        self.dataset_location = dataset_location
        self.sampling_period = sampling_period
        self.df_location = dataset_location + str(dataset_index).zfill(2) + "_tracks.csv"
        self.static_info_location = dataset_location + str(dataset_index).zfill(2) + "_tracksMeta.csv"
        self.video_info_location = dataset_location + str(dataset_index).zfill(2) + "_recordingMeta.csv"
        
        self.bg_image = None
        self.bg_image_scaling_factor = None
        self.total_frame_number = 0
        self.longest_trajectory = 0
        self.historical_aggregations = None

        self.frame_step = int(np.floor(self.sampling_period / 40))

        self.df = pd.read_csv(self.df_location)
        self.static_info = pd.read_csv(self.static_info_location)
        self.video_info = read_meta_info(self.video_info_location)
        self.frame_delay = 1000 / self.video_info["frameRate"]

        self.df_list, self.df_direction_list = self.get_df_list()

        self.get_background_img(dataset_location + str(dataset_index).zfill(2) + "_highway.png")

    def get_background_img(self, path):
        '''
        Sets the background image for plotting and creating gifs.
            Parameters:
                path (str): path to the png file that contains the background image
            Returns:
                Nothing
        '''
        self.bg_image = plt.imread(path)
        self.bg_image_scaling_factor = np.max((np.max(self.bg_image.shape), self.longest_trajectory)) / np.min((np.max(self.bg_image.shape), self.longest_trajectory))  # Calculates the scaling factor between img size and trajectory length

    def get_df_list(self):
        """
        Get data organized into lists by frame and driving direction.

        Returns:
            tuple: A tuple containing two lists.
                - List 1: DataFrames grouped by frame.
                - List 2: DataFrames grouped by frame and driving direction.
        """
        # Filter the DataFrame to include vehicles in specified frames
        frame_df = self.df.groupby((self.df.frame - 1) % self.frame_step == 0).get_group(1)

        # Calculate the total number of frames
        self.total_frame_number = frame_df.frame.nunique()

        # Merge the filtered DataFrame with static information
        merged_df = frame_df.merge(self.static_info, on='id', how='left')
        # Add a 'drivingDirection' column with initial values set to None
        frame_df['drivingDirection'] = [None] * len(frame_df)
        # Populate the 'drivingDirection' column based on 'merged_df'
        frame_df['drivingDirection'] = np.where(merged_df["drivingDirection"] == 2.0, "East", "West")
        #Refactor frame column with new sampling period
        frame_df.frame = (frame_df.frame-1) / self.frame_step
        frame_df.frame = frame_df.frame.astype(int)
        # Group vehicles by frame
        frame_groups = frame_df.groupby("frame")

        # Initialize lists to store grouped data
        self.df_list = [None] * self.total_frame_number
        self.df_direction_list = [None] * self.total_frame_number
        # Iterate through frame groups and group by driving direction
        for idx, (_, group) in enumerate(frame_groups):
            self.df_list[idx] = group
            dir_groups = group.groupby("drivingDirection")
            # Check if "East" and "West" groups exist, create empty DataFrames if not
            east_group = dir_groups.get_group("East") if "East" in dir_groups.groups else pd.DataFrame(columns=group.columns)
            west_group = dir_groups.get_group("West") if "West" in dir_groups.groups else pd.DataFrame(columns=group.columns)
            # Assign groups to the direction list
            self.df_direction_list[idx] = [east_group, west_group]
        # Calculate the longest traveled distance from static information
        self.longest_trajectory = self.static_info["traveledDistance"].max()

        return self.df_list, self.df_direction_list

    def get_historical_vehicle_aggregations(self, bubble_radius=50.0, lookback_window=2, frame_step=1):
        """
        Get historical vehicle aggregations for each ego vehicle in frames.

        Args:
            bubble_radius (float): Radius of the bubble around the ego vehicle (default: 50 meters).
            lookback_window (int): Number of frames to look back (default: 2).
            frame_step (int): Spacing between frame indices (default: 1).

        Returns:
            dataframe: A dataframe, containing every historical vehicle aggregation identified with an index.
        """
        def in_bubble(ego_vehicle, x, radius = bubble_radius):
            '''
            Gets the distance between two vehicles.
            Notation: 
                (x, y) - coordinates of the top left corner of the bounding box of the vehicle.
                (w, h) - width and height of the bounding box.

                Parameters:
                    ego_vehicle: dataframe row with information about the ego vehicle
                    x: dataframe row with information about the vehicle to consider
                Returns:
                    dist: distance between vehicles
            '''

            x1, y1, w1, h1 = ego_vehicle.x, ego_vehicle.y, ego_vehicle.width, ego_vehicle.height
            x2, y2, w2, h2 = x.x, x.y, x.width, x.height
            c1 = np.array([x1 + w1/2, y1 + h1/2])
            c2 = np.array([x2 + w2/2, y2 + h2/2])
            dist = np.linalg.norm(c1-c2)

            return dist < radius
        
        eastbound_frames = [direction[0] for direction in self.df_direction_list]
        westbound_frames = [direction[1] for direction in self.df_direction_list]
        historical_aggregations = pd.DataFrame(columns = eastbound_frames[0].columns)
        historical_aggregation_index = 1
        
        for frame_df in eastbound_frames[lookback_window::frame_step]:
            if len(frame_df) < 1: continue
            idx = frame_df.frame.iloc[0]
            past_frames_idx = [idx-i for i in range(lookback_window+1)]
            for _, ego_vehicle in frame_df.iterrows():
                historical_group = pd.DataFrame(columns=frame_df.columns)
                for iter_frame in [eastbound_frames[i] for i in past_frames_idx[::-1]]:
                    if ego_vehicle.id in iter_frame.id.tolist(): #If the ego vehicle is not in previous frame, use current position
                        ego_vehicle = iter_frame[iter_frame.id == ego_vehicle.id].iloc[0]
                    iter_frame = iter_frame.drop(iter_frame.index[iter_frame["id"] == ego_vehicle.id].tolist(), axis = 'index') #drop ego vehicle from calculations
                    mask = iter_frame.apply((lambda x: in_bubble(ego_vehicle, x, radius=50)), axis = 1)
                    historical_group = pd.concat([historical_group, pd.DataFrame(ego_vehicle).T, iter_frame[mask]], axis = 0)
                historical_group['historical_aggregation_index'] = historical_aggregation_index
                historical_aggregation_index += 1
                historical_aggregations = pd.concat([historical_aggregations, historical_group])

        for frame_df in westbound_frames[lookback_window::frame_step]:
            if len(frame_df) < 1: continue
            idx = frame_df.frame.iloc[0]
            past_frames_idx = [idx-i for i in range(lookback_window+1)]
            for _, ego_vehicle in frame_df.iterrows():
                historical_group = pd.DataFrame(columns=frame_df.columns)
                for iter_frame in [eastbound_frames[i] for i in past_frames_idx[::-1]]:
                    if ego_vehicle.id in iter_frame.id.tolist(): #If the ego vehicle is not in previous frame, use current position
                        ego_vehicle = iter_frame[iter_frame.id == ego_vehicle.id].iloc[0]
                    iter_frame = iter_frame.drop(iter_frame.index[iter_frame["id"] == ego_vehicle.id].tolist(), axis = 'index') #drop ego vehicle from calculations
                    mask = iter_frame.apply((lambda x: in_bubble(ego_vehicle, x, radius=50)), axis = 1)
                    historical_group = pd.concat([historical_group, pd.DataFrame(ego_vehicle).T, iter_frame[mask]], axis = 0)
                historical_group['historical_aggregation_index'] = historical_aggregation_index
                historical_aggregation_index += 1
                historical_aggregations = pd.concat([historical_aggregations, historical_group])

        self.historical_aggregations = historical_aggregations.astype({"historical_aggregation_index": int})
        return self.historical_aggregations

    def save_vehicle_groups(self, path):
        """
        Saves the vehicle groups that were extracted from frame information into a csv file in the location of the dataset.
            Parameters:
                path: directory in which to save the csv file of the dataframe
            Returns:
                None.
        """
        file_path = path + "/" + str(self.dataset_index).zfill(2) + "_groups.csv"
        if os.path.exists(file_path): #if the file already exists, we need to erase it to avoid duplicity
            print("File " + file_path + " already exists. Deleting...")
            os.remove(file_path)
        print(f"Saving to: {file_path}")
        with open(file_path, 'a') as f: #write the dataframes on the corresponding file
            self.historical_aggregations.to_csv(f, index=True, index_label='Index')
        
    
    def plot_frame(self, frame_number=None):
        '''
        Visualization function. Plots the bounding boxes of every vehicle in a determined frame.

            Parameters:
                Frame number: frame to plot. If none is given, chosen at random
            Returns:
                Nothing.
        '''
        if frame_number is None:
            frame_number = np.random.randint(0, self.total_frame_number)

        eastbound_df, westbound_df = self.df_direction_list[frame_number]
        fig, ax = plt.subplots()
        if self.bg_image is not None:
            ax.imshow(self.bg_image)

        for index, vehicle in eastbound_df.iterrows():
            rect = Rectangle(
                (int(vehicle.x * self.bg_image_scaling_factor), int(vehicle.y * self.bg_image_scaling_factor)), int(vehicle.width * self.bg_image_scaling_factor), int(vehicle.height * self.bg_image_scaling_factor),
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)
        for index, vehicle in westbound_df.iterrows():
            rect = Rectangle(
                (int(vehicle.x * self.bg_image_scaling_factor), int(vehicle.y * self.bg_image_scaling_factor)), int(vehicle.width * self.bg_image_scaling_factor), int(vehicle.height * self.bg_image_scaling_factor),
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)
        plt.show()

    def plot_frame_anim(self, frame_number, ax):
        '''
        Visualization function. Plots the bounding boxes of every vehicle in a determined frame. Specifically created for animating a .gif

            Parameters:
                Frame number: frame to plot. Provided by the FuncAnimation class in matplotlib.
                ax: axis in which to plot the information. 
            Returns:
                Nothing.
        '''
        ax.clear()
        ax.imshow(self.bg_image)
        eastbound_df, westbound_df = self.df_direction_list[frame_number]
        for index, vehicle in eastbound_df.iterrows():
            rect = Rectangle(
                (int(vehicle.x * self.bg_image_scaling_factor), int(vehicle.y * self.bg_image_scaling_factor)), int(vehicle.width * self.bg_image_scaling_factor), int(vehicle.height * self.bg_image_scaling_factor),
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)
        for index, vehicle in westbound_df.iterrows():
            rect = Rectangle(
                (int(vehicle.x * self.bg_image_scaling_factor), int(vehicle.y * self.bg_image_scaling_factor)), int(vehicle.width * self.bg_image_scaling_factor), int(vehicle.height * self.bg_image_scaling_factor),
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)

    def plot_group(self, group_number = None):
        '''
        Visualization function. Plots the bounding boxes of every vehicle in a determined frame.

            Parameters:
                Group number: group to plot. If none is given, chosen at random
            Returns:
                Nothing.
        '''
        if group_number is None:
            group_number = np.random.randint(0, 2*self.total_frame_number)
        group_df = self.vehicle_groups[group_number]

        fig, ax = plt.subplots()
        if self.bg_image is not None:
            ax.imshow(self.bg_image)

        for index, vehicle in group_df.iterrows():
            rect = Rectangle(
                (int(vehicle.x * self.bg_image_scaling_factor), int(vehicle.y * self.bg_image_scaling_factor)), int(vehicle.width * self.bg_image_scaling_factor), int(vehicle.height * self.bg_image_scaling_factor),
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)
        plt.show()

    def create_gif(self, save_animation=False, frames=None, speed_up_factor=1):
        fig, ax = plt.subplots()
        fig.set_tight_layout(True)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(self.bg_image)
        self.anim = animation.FuncAnimation(
            fig=fig,
            func=self.plot_frame_anim,
            fargs=(ax,),
            frames=frames if frames is not None else self.total_frame_number,
            interval=self.frame_delay / speed_up_factor,
            blit=False,
        )
        # HTML(anim.to_html5_video())

        if save_animation:
            writer = animation.PillowWriter(
                fps=self.video_info["frameRate"] * speed_up_factor,
                metadata=dict(artist="Me"),
                bitrate=900,
            )
            self.anim.save("test.gif", writer=writer)
        # plt.show()
        return self.anim
 
def main():
    if platform == 'darwin':
        dataset_location = "/Users/lmiguelmartinez/Tesis/datasets/highD/data/"
    else:
        dataset_location = "/home/lmmartinez/Tesis/datasets/highD/data/"
    sampling_period = 1000
    df = pd.DataFrame(columns=['Loading', 'Get groups', 'Save groups', 'Frames', 'Groups', 'Bubble radius', 'Frame Step', 'Frame Lookback'])

    row_list = []
    lookback_window = 8
    frame_step = 8
    bubble_radius = 50

    for dataset_index in range(1,61):
        print("Loading data - File " + str(dataset_index).zfill(2))
        p1 = time()
        scene_data = SceneData(dataset_location=dataset_location, dataset_index=dataset_index, sampling_period= 1000)
        p2 = time()
        print("Data loaded - Time elapsed is: {} seconds".format(p2-p1))
        print("Total number of frames is: {}".format(scene_data.total_frame_number))
        print("Extracting groups")
        p3 = time()
        vehicle_groups = scene_data.get_historical_vehicle_aggregations(bubble_radius=bubble_radius, lookback_window=4, frame_step=8)
        p4 = time()
        print("Groups extracted - Time elapsed is: {} seconds".format(p4-p3))
        print("Saving to csv")
        p5 = time()
        scene_data.save_vehicle_groups(path='/home/lmmartinez/Tesis/datasets/highD/groups_historic_1000ms')
        p6 = time()
        print("Groups saved - Time elapsed is: {} seconds".format(p6-p5))
        print("Total number of groups is: {}".format(len(vehicle_groups.groupby('historical_aggregation_index'))))
        print("")
        print("______________________________________________")
        print("")
        row_list.append([p2-p1, p4-p3, p6-p5, scene_data.total_frame_number, len(vehicle_groups.groupby('historical_aggregation_index')), bubble_radius, frame_step, lookback_window])

    df = pd.concat([df, pd.DataFrame(row_list, columns = df.columns)])
    df.to_csv(dataset_location + 'computation_info_' + str(sampling_period) + 'ms.csv', index=True, index_label='File')

if __name__ == "__main__":
    main()
