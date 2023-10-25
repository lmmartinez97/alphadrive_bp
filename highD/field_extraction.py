import h5py
import numpy as np
import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
import os
import seaborn as sns

from rich import print
from sys import platform
from time import time

from read_csv import read_groups

sns.set()
sns.set_style("whitegrid")
sns.set_context("paper")
sns.color_palette("hls", 8)

class PotentialField:
    '''
    A class that represents the occupation information of a frame - See SceneData in dataviz.py.

    Attributes:
        radius_x, radius_y: semiaxis of the ellipse that is considered around the ego vehicle
        step_x, step_y: space discretization in each axis
        df_group_list: a list of vehicle grups as imported by the function read_groups of read_csv.py
    '''
    def __init__(self, radius_x, radius_y, step_x, step_y, df_group_list):
        self.rx, self.ry = radius_x, radius_y
        self.sx, self.sy = step_x, step_y
        self.df_group_list = df_group_list

        self.grid = self.initialize_grid()
        self.field_list = [None] * len(self.df_group_list)

    def initialize_grid(self):
        '''
        Initializes a potential field grid and calculates the number of pixels that the final image will have.
        
        Parameters:
                None
            Returns:
                A grid initialized to zero with size (num_y, num_x)
        '''
        #calculate grid size
        self.num_x = int(2*self.rx / self.sx) + 1
        self.num_y = int(2*self.ry / self.sy) + 1
        self.grid = np.zeros([self.num_y, self.num_x], dtype = np.float32) #x dimension is rows, y dimension is columns
        self.x_pos, self.y_pos = np.linspace(-self.rx, self.rx, self.num_x), np.linspace(-self.ry, self.ry, self.num_y) #x and y interchanged so x is the horizontal dimension, and y is the vertical dimension
        print("Initialized grid of shape: {}".format(self.grid.shape))
        return self.grid

    def group_to_list(self, group):
        '''
        Converts the group from dataframe to dictionary and casts every value to float for future manipulation.
        
        Parameters:
                None
            Returns:
                A list of dictionaries in which each element contains the information of an entry of the dataframe. Order is preserved.
        '''

        self.vehicle_list = group.to_dict(orient='records')
        for idx, vehicle in enumerate(self.vehicle_list):
            for k, v in vehicle.items():
                self.vehicle_list[idx][k] = float(v)
        return self.vehicle_list

    def get_field_value(self, x, y, a, b, c, k):
        '''
        Calculates the value of a potential field along two linespace inputs: x and y coordinates. 
        See https://stackoverflow.com/a/22778484 for magic around empty dimensions - I still don't understand how, but it works.
        
        Parameters:
                x, y: linespaces along which the field is calculated. Positions are relative to the vehicle that is being considered, 
                      which is itself relative to the ego vehicle
                k, a, b, c: parameters of the gaussian - amplitude and exponent coefficients. 
                            The calculation of a, b and c is done from standard deviations and an angle of rotation
            Returns:
                A numpy array of size (num_y, num_x) in which each element is the value of the potential field evaluated 
                in the corresponding coordinate of the input linespaces
        '''
        local_grid = k * np.exp(-(a * np.square(x[None,:]) + 2 * b * x[None,:] * y[:,None] + c * np.square(y[:,None])))
        return local_grid
    
    def calculate_field(self):
        '''
        Calculates the value of a potential field associated to an ego vehicle in the context of its group
        
        Parameters:
                None
            Returns:
                A numpy array of size (num_y, num_x) in which each element is the value of the potential field evaluated 
                in the corresponding coordinate of the input linespaces
        '''
        self.grid = np.zeros([self.num_y, self.num_x], dtype = np.float32) #x dimension is rows, y dimension is columns
        for vehicle in self.vehicle_list:
            x_rel = vehicle["x"] - self.x_ego
            y_rel = - vehicle["y"] + self.y_ego #positive y direction points "downwards"
            dx, dy = self.x_pos - x_rel, self.y_pos - y_rel
            ang = np.arctan2(vehicle["xVelocity"], vehicle["yVelocity"])
            sigma_y, sigma_x = vehicle["width"], vehicle["height"]/2
            a = np.square(np.cos(ang)) / (2 * np.square(sigma_x)) + np.square(np.sin(ang)) / (2 * np.square(sigma_y))
            b = - np.sin(2 * ang) / (4 * np.square(sigma_x)) + np.sin(2 * ang) / (4 * np.square(sigma_y))
            c = np.square(np.sin(ang)) / (2 * np.square(sigma_x)) + np.square(np.cos(ang)) / (2 * np.square(sigma_y))
            k = np.linalg.norm([vehicle["xVelocity"], vehicle["yVelocity"]])
            self.grid = np.add(self.grid, self.get_field_value(dx, dy, a, b, c, k))

        #normalize potential field
        self.normalized_grid = self.grid / np.linalg.norm(self.grid, 'fro')
        self.normalized_grid = (self.normalized_grid - np.min(self.normalized_grid)) / (np.max(self.normalized_grid) - np.min(self.normalized_grid))
        return self.normalized_grid
    
    def calculate_field_list(self):
        '''
        Performs potential field calculations for every group in df_group_list
        
        Parameters:
                None
            Returns:
                A list in which each element is the potential field representation of the associated group.
        '''
        for idx, group in enumerate(self.df_group_list):
            if not (idx%1000):
                print("Processing group {} of {}".format(idx, len(self.df_group_list)))
            self.vehicle_list = self.group_to_list(group)
            self.vehicle_count = len(self.vehicle_list)
            self.ego_vehicle = self.vehicle_list[0]
            self.x_ego, self.y_ego = self.ego_vehicle["x"], self.ego_vehicle["y"]
            self.field_list[idx] = self.calculate_field()

        return self.field_list
    
    def plot_field(self, idx = None):
        '''
        Plots a 3d graph and a heat map of the potential field of a given group in the variable idx.
        
        Parameters:
                idx: index of the group that will be represented
            Returns:
                None
        '''

        if idx is None:
            idx = np.random.randint(len(self.field_list))
        fig = plt.figure(figsize=(15, 8))

        #plot 3d field
        ax3d = fig.add_subplot(1, 2, 1, projection='3d')
        X, Y = np.meshgrid(self.x_pos, self.y_pos)
        surf = ax3d.plot_surface(X, Y, self.field_list[idx], cmap = "viridis")
        ax3d.set_xlabel("Longitudinal axis (m)")
        ax3d.set_ylabel("Transversal axis (m)")
        ax3d.set_zlabel("Potential field magnitude (-)")
        ax3d.set_title("3d plot - Normalized between 0 and 1")
        fig.colorbar(surf, orientation = 'horizontal', pad = 0.2)

        #plot heatmap
        ax2d = fig.add_subplot(1, 2, 2)
        img = ax2d.imshow(self.field_list[idx])
        ax2d.set_xlabel("Longitudinal axis (px)")
        ax2d.set_ylabel("Transversal axis (px)")
        ax2d.set_title("Potential field heat map")
        fig.colorbar(img, orientation = 'horizontal', pad = 0.2)

        fig.suptitle("Potential field representation")
        plt.show()

    def plot_center_field(self, idx):
        '''
        Plots the value of the potential field in the longitudinal axis of the ego vehicle of a given group in the variable idx.
        
        Parameters:
                idx: index of the group that will be represented
            Returns:
                None
        '''
        if idx is None:
            idx = np.random.randint(len(self.field_list))
        fig, ax = plt.subplots()
        y_index = int((self.field_list[idx].shape[0] - 1) / 2)
        x = self.x_pos
        y = self.self.field_list[idx][y_index]
        ax.plot(x, y)
        ax.set_xlabel("Transversal axis (m)")
        ax.set_ylabel("Potential field magnitude (-)")
        ax.set_title("Ego vehicle longitudinal axis - Frame {}".format(int(self.ego_vehicle["frame"])))

        plt.show()

    def plot_heat_map(self):
        '''
        Plots the heat map of the potential field of a given group in the variable idx.
        
        Parameters:
                idx: index of the group that will be represented
            Returns:
                None
        '''
        if idx is None:
            idx = np.random.randint(len(self.field_list))
        fig, ax = plt.subplots()
        ax.imshow(self.self.field_list[idx])
        ax.set_xlabel("Longitudinal axis (m)")
        ax.set_ylabel("Transversal axis (m)")
        ax.set_title("Potential field heat map - Frame {}".format(int(self.ego_vehicle["frame"])))
        plt.show()

def split_and_save_to_hdf5(image_list, train_ratio, val_ratio, hdf5_file):
    """
    Split a list of NumPy arrays into train, validation, and test datasets and save them to an HDF5 file.

    :param image_list: List of NumPy arrays (images).
    :param train_ratio: Ratio of data to allocate for training (e.g., 0.7 for 70%).
    :param val_ratio: Ratio of data to allocate for validation (e.g., 0.15 for 15%).
    :param hdf5_file: Name of the HDF5 file to save the datasets.
    """
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("The sum of train_ratio and val_ratio should be less than 1.0.")

    total_samples = len(image_list)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    test_size = total_samples - train_size - val_size

    # Shuffle the image list to randomize the data
    np.random.shuffle(image_list)

    # Split the data into train, validation, and test sets
    train_data = image_list[:train_size]
    val_data = image_list[train_size:train_size + val_size]
    test_data = image_list[train_size + val_size:]

    print("Saving to file " + hdf5_file)
    # Save the datasets to an HDF5 file
    with h5py.File(hdf5_file, "w") as hf:
        hf.create_dataset("train", data=np.array(train_data))
        hf.create_dataset("validation", data=np.array(val_data))
        hf.create_dataset("test", data=np.array(test_data))

def main():
    if platform == 'darwin':
        dataset_location = "/Users/lmiguelmartinez/Tesis/datasets/highD/groups_1000ms/"
    else:
        dataset_location = "/home/lmmartinez/Tesis/datasets/highD/groups_1000ms/"
    rx = 50 # horizontal semiaxis of ellipse to consider as ROI
    ry = 6 # vertical semiaxis of ellipse to consider as ROI
    sx = 0.5
    sy = 0.1

    for dataset_index in range(1,61):
        print("Importing group list {} of 60".format(dataset_index))
        df_groups_list = read_groups(dataset_location + str(dataset_index).zfill(2) + "_groups.csv")
        start = time()
        field = PotentialField(rx, ry, sx, sy, df_groups_list)
        field_list = field.calculate_field_list()
        end = time()
        print("Time taken is {}".format(end-start))
        print("The number of images generated is: {}".format(len(field_list)))
        filename = '/home/lmmartinez/OneDrive/Tesis/processed_datasets/images_1000ms' + str(dataset_index).zfill(2) + '_1000ms.hdf5'
        if os.path.exists(filename):
            try:
                os.remove(filename)
                print(f"File '{filename}' has been deleted.")
            except OSError as e:
                print(f"Error deleting the file '{filename}': {e}")
        else:
            # File does not exist, continue with the program
            print(f"File '{filename}' does not exist.")
            split_and_save_to_hdf5(image_list=field_list, train_ratio=0.6, val_ratio=0.2, hdf5_file=filename)

if __name__ == "__main__":
    main()