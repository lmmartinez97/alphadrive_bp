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


def print_bl():
    print("\n")


mlp.rcParams["animation.embed_limit"] = (
    2**128
)  # Increased animation size to save gifs if necessary

class SceneData:
    '''
    A class that represents an entire recording of the highD dataset.

    Attributes:
        dataset_location: a path to the directory in which the dataset is stored.
        dataset_index: index of the recording that is to be addressed
    '''
    def __init__(self, dataset_location = None, dataset_index = None):
        self.dataset_index  = dataset_index
        self.dataset_location = dataset_location
        self.df_location = dataset_location + str(dataset_index).zfill(2) + "_tracks.csv"
        self.static_info_location = dataset_location + str(dataset_index).zfill(2) + "_tracksMeta.csv"
        self.video_info_location = dataset_location + str(dataset_index).zfill(2) + "_recordingMeta.csv"
        self.df_location = dataset_location + str(dataset_index).zfill(2) + "_tracks.csv"

        self.df = pd.read_csv(self.df_location)
        self.static_info = read_static_info(self.static_info_location)
        self.video_info = read_meta_info(self.video_info_location)
        self.frame_delay = 1000 / self.video_info["frameRate"]

        self.df_list = self.get_df_list()
        self.total_frame_number = len(self.df_list)
        self.longest_trajectory = 0
        self.df_direction_list = self.get_df_direction_list()
        self.bg_image = None
        self.bg_image_scaling_factor = None
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

    def get_distance_between_vehicles(self, t1, t2):
        '''
        Gets the distance between two vehicles.
        Notation: 
            (x, y) - coordinates of the top left corner of the bounding box of the vehicle.
            (w, h) - width and height of the bounding box.

            Parameters:
                t1: Tuple-like in the form (x1, y1, w1, h1).
                t2: Tuple-like in the form (x2, y2, w2, h2)
            Returns:
                dist: distance between vehicles
        '''
        x1, y1, w1, h1 = t1
        x2, y2, w2, h2 = t2
        c1 = np.array([x1 + w1/2, y1 + h1/2])
        c2 = np.array([x2 + w2/2, y2 + h2/2])
        dist = np.linalg.norm(c1-c2)
        return dist
    
    def get_df_list(self):
        '''
        Refactors information from track format to frame format.
            Parameters:
                None, inputs to the method are attributes of the class
            Returns:
                df_list: a list of dataframes - one per frame of recorded data - in which every vehicle in the scene is included.
        '''
        self.df_list = [None] * self.df.frame.nunique()  # Preallocate list to improve speed - number of dataframes is equal to that of video frames
        for frame_number in range(1, self.df.frame.nunique() + 1, 1):  # Iterate through every video frame in the dataframe - frame enumeration starts at 1
            self.df_list[frame_number - 1] = self.df[np.in1d(self.df["frame"].values, [frame_number])]  # Select rows of dataframe with df["frame"] == frame_number
        return self.df_list

    def get_df_direction_list(self):
        '''
        Separates vehicles in frames according to the direction they are driving in.
            Parameters:
                None, inputs to the method are attributes of the class
            Returns:
                df_direction_list: a list of tuples - one tuple per every frame, with two elements: eastbound and westbound vehicles dataframes -.
        '''
        eastbound_vehicles = []  # contains the vehicle_id of every vehicle driving East
        westbound_vehicles = []  # contains the vehicle_id of every vehicle driving West
        for vehicle_id in self.static_info.keys():
            self.longest_trajectory = (self.static_info[vehicle_id]["traveledDistance"] if self.longest_trajectory < self.static_info[vehicle_id]["traveledDistance"] else self.longest_trajectory)  # get longest trajectory in dataset to calculate img scale factor
            if self.static_info[vehicle_id]["drivingDirection"] == 2.0:
                eastbound_vehicles.append(vehicle_id)
            else:
                westbound_vehicles.append(vehicle_id)

        self.df_direction_list = [None] * self.df.frame.nunique()  # For every frame, save two dataframes - one for eastbound vehicles and one for westbound vehicles
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

    def get_vehicle_groups_from_frame(self, frame):
        '''
        Given a frame, it creates a list in which every element is a dataframe with an ego vehicle and all those vehicles that are less closer than a predetermined distance away.
        The length of the list is the number of vehicles that are present in the frame.

        TODO: dynamic distance threshold - set to 50m at the time.
            Parameters:
                Frame: a dataframe containing information of a specific frame - see get_df_list and get_df_direction_list for details
            Returns:
                group_df_list: a list in which every element contains a vehicle group in the dataframe form
        '''
        group_df_list = [None]*len(frame) #for every vehicle in the frame we will save a list of dataframes in which a vehicle group is represented
        drop_index_list = []

        for i ,(index, vehicle_id) in enumerate(frame.iterrows()): #iterate through every vehicle in the frame - returns index of entry and pd.Series with entry
            t1 = (vehicle_id.x, vehicle_id.y, vehicle_id.width, vehicle_id.height) #get position data of ego vehicle
            new_frame = frame.drop([index], axis = 0) #drop ego vehicle to not compare against itself
            drop_index_list.clear() #clear drop index list - it is faster to allocate it outside the loop than to create it in every iteration
            for index_local,vehicle_id_local in new_frame.iterrows(): #compare the ego vehicle against every vehicle in frame
                t2 = (vehicle_id_local.x, vehicle_id_local.y, vehicle_id_local.width, vehicle_id_local.height) #get position data of other vehicles
                if self.get_distance_between_vehicles(t1, t2) > 50: #if other vehicle is more than 50m away
                    drop_index_list.append(index_local) #store vehicles that are further than 50 m away
            new_frame = frame.drop(drop_index_list, axis = 0) #drop vehicles from dataframe
            index_list = list(new_frame.index) #get indices that were not dropped in a list
            index_list.remove(index) #remove the ego vehicle
            index_list.insert(0, index) #insert the ego vehicle in first place
            new_frame = new_frame.loc[index_list] #reorder the dataframe
            group_df_list[i] = new_frame #store filtered dataframe in list of frame groups

        return group_df_list
    
    def get_vehicle_groups(self):
        '''
        Given a list of frames, it creates the vehicle groups associated to them by calling get_vehicle_groups_from_frame.

            Parameters:
                None, inputs to this method are attributes of the class
            Returns:
                vehicle groups: a list in which every element is another list. The elements of the inner list are dataframes in which the information of a vehicle group is stored
        '''
        self.vehicle_groups = [None] * 2 * self.total_frame_number #preallocate for speed
        #get vehicle groups of eastbound vehicles
        for i in range(self.total_frame_number):
            frame = self.df_direction_list[i][0]
            self.vehicle_groups[i] = self.get_vehicle_groups_from_frame(frame)
            frame = self.df_direction_list[i][1]
            self.vehicle_groups[i+self.total_frame_number] = self.get_vehicle_groups_from_frame(frame)
        return self.vehicle_groups
    
    def save_vehicle_groups(self):
        """
        Saves the vehicle groups that were extracted from frame information into a csv file in the location of the dataset.

            Parameters:
                None, inputs to this method are attributes of the class.
            Returns:
                None.
        """
        file_path = self.dataset_location + str(self.dataset_index).zfill(2) + "_groups.csv"
        if os.path.exists(file_path): #if the file already exists, we need to erase it to avoid duplicity
            print("File " + file_path + " already exists. Deleting...")
            os.remove(file_path)
        self.vehicle_groups_flat = [group for frame in self.vehicle_groups for group in frame] #flatten the vehicle groups list
        with open(file_path, 'a') as f: #write the dataframes on the corresponding file
            for df in self.vehicle_groups_flat:
                df.to_csv(f, index=True, index_label='Index')
                f.write("\n")
        
    
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
                (
                    int(vehicle.x * self.bg_image_scaling_factor),
                    int(vehicle.y * self.bg_image_scaling_factor),
                ),
                int(vehicle.width * self.bg_image_scaling_factor),
                int(vehicle.height * self.bg_image_scaling_factor),
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)
        for index, vehicle in westbound_df.iterrows():
            rect = Rectangle(
                (
                    int(vehicle.x * self.bg_image_scaling_factor), #x
                    int(vehicle.y * self.bg_image_scaling_factor), #y
                ),
                int(vehicle.width * self.bg_image_scaling_factor), #width
                int(vehicle.height * self.bg_image_scaling_factor), #height
                linewidth=1,
                edgecolor="g",
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
                (
                    int(vehicle.x * self.bg_image_scaling_factor),
                    int(vehicle.y * self.bg_image_scaling_factor),
                ),
                int(vehicle.width * self.bg_image_scaling_factor),
                int(vehicle.height * self.bg_image_scaling_factor),
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)
        for index, vehicle in westbound_df.iterrows():
            rect = Rectangle(
                (
                    int(vehicle.x * self.bg_image_scaling_factor),
                    int(vehicle.y * self.bg_image_scaling_factor),
                ),
                int(vehicle.width * self.bg_image_scaling_factor),
                int(vehicle.height * self.bg_image_scaling_factor),
                linewidth=1,
                edgecolor="g",
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
                (
                    int(vehicle.x * self.bg_image_scaling_factor),
                    int(vehicle.y * self.bg_image_scaling_factor),
                ),
                int(vehicle.width * self.bg_image_scaling_factor),
                int(vehicle.height * self.bg_image_scaling_factor),
                linewidth=1,
                edgecolor="b",
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
                fps=scene_data.video_info["frameRate"] * speed_up_factor,
                metadata=dict(artist="Me"),
                bitrate=900,
            )
            anim.save("test.gif", writer=writer)
        # plt.show()
        return self.anim
   
def main():
    if platform == 'darwin':
        dataset_location = "/Users/lmiguelmartinez/Tesis/datasets/highD/data/"
    else:
        dataset_location = "/home/lmmartinez/Tesis/datasets/highD/data/"

    df = pd.DataFrame(columns=['Loading', 'Get groups', 'Save groups', 'Frames', 'Groups'])

    row_list = []

    for dataset_index in range(1,61):
        print("Loading data - File " + str(dataset_index).zfill(2))
        p1 = time()
        scene_data = SceneData(dataset_location=dataset_location, dataset_index=dataset_index)
        p2 = time()
        print("Data loaded - Time elapsed is: {} seconds".format(p2-p1))
        print("Total number of frames is: {}".format(scene_data.total_frame_number))
        print("Extracting groups")
        p3 = time()
        vehicle_groups = scene_data.get_vehicle_groups()
        p4 = time()
        print("Groups extracted - Time elapsed is: {} seconds".format(p4-p3))
        print("Total number of groups is: {}".format(len(scene_data.vehicle_groups_flat)))
        print("Saving to csv")
        p5 = time()
        scene_data.save_vehicle_groups()
        p6 = time()
        print("Groups saved - Time elapsed is: {} seconds".format(p5-p6))
        row_list.append([p2-p1, p4-p3, p6-p5, scene_data.total_frame_number, len(scene_data.vehicle_groups_flat)])

    df = pd.concat([df, pd.DataFrame(row_list, columns = df.columns)])
    df.to_csv(dataset_location + 'computation_info.csv', index=True, index_label='File')

if __name__ == "__main__":
    main()
