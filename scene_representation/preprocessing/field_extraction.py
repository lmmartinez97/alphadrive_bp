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

import tqdm


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

        self.num_x = int(2*self.rx / self.sx) + 1
        self.num_y = int(2*self.ry / self.sy) + 1
        self.grid = np.zeros([self.num_y, self.num_x], dtype = np.float32) #x dimension is rows, y dimension is columns
        self.x_pos, self.y_pos = np.linspace(-self.rx, self.rx, self.num_x), np.linspace(-self.ry, self.ry, self.num_y) #x and y interchanged so x is the horizontal dimension, and y is the vertical dimension
        print("Initialized grid of shape: {}".format(self.grid.shape))
    
    def calculate_field(self, inner_group, ego_vehicle):
        '''
        Calculates the value of a potential field associated to an ego vehicle in the context of its group in a particular frame
        
        Parameters:
                None
            Returns:
                A numpy array of size (num_y, num_x) in which each element is the value of the potential field evaluated 
                in the corresponding coordinate of the input linespaces
        '''
        def get_field_value(x, y, a, b, c, k):
            '''
            Calculates the value of a potential field along two linespace inputs: x and y coordinates. 
            See https://stackoverflow.com/a/22778484 for magic around empty dimensions - I still don't understand how, but it works.
            
            Parameters:
                    x, y: linespaces along which the field is calculated. Positions are relative to the vehicle that is being considered, 
                        which is itself relative to the ego vehicle
                    k, a, b, c: parameters of the gaussian - amplitude and exponent coefficients. 
                                The calculation of a, b and c is done from standard deviations and angle of rotation
                Returns:
                    A numpy array of size (num_y, num_x) in which each element is the value of the potential field evaluated 
                    in the corresponding coordinate of the input linespaces
            '''
            local_grid = k * np.exp(-(a * np.square(x[None,:]) + 2 * b * x[None,:] * y[:,None] + c * np.square(y[:,None])))
            return local_grid
        
        def calculate_parameters_and_field_value(ego_vehicle, vehicle):
            '''
            Calculates the parameters of the potential field associated to a particular vehicle in the context of a group.
            Serves as a function call to the apply method of the dataframe that contains the information of the vehicles in the group in a particular frame.
            
            Parameters:
                    ego_vehicle: instance of dataframe that contains the information of the ego vehicle in a particular frame.
                    vehicle: vehicle which field is beign considered
                Returns:
            '''
            x1, y1 = ego_vehicle.x, ego_vehicle.y
            x2, y2, vx2, vy2, w2, h2 = vehicle.x, vehicle.y, vehicle.xVelocity, vehicle.yVelocity, vehicle.width, vehicle.height
            x_rel = x2 - x1
            y_rel = - y2 + y1 #positive y direction points "downwards"
            dx, dy = self.x_pos - x_rel, self.y_pos - y_rel
            ang = np.arctan2(vx2, vy2)
            sigma_y, sigma_x = w2, h2/2
            a = np.square(np.cos(ang)) / (2 * np.square(sigma_x)) + np.square(np.sin(ang)) / (2 * np.square(sigma_y))
            b = - np.sin(2 * ang) / (4 * np.square(sigma_x)) + np.sin(2 * ang) / (4 * np.square(sigma_y))
            c = np.square(np.sin(ang)) / (2 * np.square(sigma_x)) + np.square(np.cos(ang)) / (2 * np.square(sigma_y))
            k = np.sqrt(np.linalg.norm((vehicle.xVelocity, vehicle.yVelocity)))
            self.grid = np.add(self.grid, get_field_value(dx, dy, a, b, c, k))
            return self.grid

        self.grid = np.zeros([self.num_y, self.num_x], dtype = np.float32) #x dimension is rows, y dimension is columns
        for _, row in inner_group.iterrows():
            self.grid += calculate_parameters_and_field_value(ego_vehicle=ego_vehicle, vehicle=row)

        return self.grid
    
    def calculate_field_list(self):
        '''
        Performs potential field calculations for every group in df_group_list
        
        Parameters:
                None
            Returns:
                An np.array in which each element is the potential field representation of the associated group.
        '''
        self.field_list = []
        for idx, (_, group) in enumerate(self.df_group_list.groupby("historical_aggregation_index")):
            if idx % 100 == 0:
                print("Processing group {} of {}".format(idx, self.df_group_list["historical_aggregation_index"].nunique()))
            group["frame"] = - group["frame"] + group["frame"].max() + 1 # determine relative frame step to atenuate distant values
            #Current frame will have a value of 1, previous frame will have a value of 2, nth frame will have a value of n
            ego_vehicle_id = group.iloc[0].id
            temp_grid = np.zeros([self.num_y, self.num_x], dtype = np.float32) #x dimension is rows, y dimension is columns
            for frame_number, inner_group in group.groupby("frame"): #add the field for every frame
                frame_attenuation = 1/inner_group.iloc[0].frame
                ego_vehicle = inner_group[inner_group["id"] == ego_vehicle_id].iloc[0]
                if frame_number != 1:
                    inner_group = inner_group[inner_group["id"] != ego_vehicle_id]
                temp_grid = np.add(self.calculate_field(inner_group, ego_vehicle) * frame_attenuation, temp_grid)
            self.field_list.append(temp_grid)
        self.field_list = np.asarray(self.field_list)

        return self.field_list
    
    def save_to_hdf5(self, train_ratio, val_ratio, hdf5_file):
        """
        Split a list of NumPy arrays into train, validation, and test datasets and save them to an HDF5 file.

        :param image_list: List of NumPy arrays (images).
        :param train_ratio: Ratio of data to allocate for training (e.g., 0.7 for 70%).
        :param val_ratio: Ratio of data to allocate for validation (e.g., 0.15 for 15%).
        :param hdf5_file: Name of the HDF5 file to save the datasets.
        """
        if train_ratio + val_ratio >= 1.0:
            raise ValueError("The sum of train_ratio and val_ratio should be less than 1.0.")
        
        total_samples = len(self.field_list)
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)

        #np.random.shuffle(field_list)
        # Split the data into train, validation, and test sets
        train_data = self.field_list[:train_size]
        val_data = self.field_list[train_size:train_size + val_size]
        test_data = self.field_list[train_size + val_size:]

        #reescale the images for saving
        train_max, train_min = np.max(train_data), np.min(train_data)

        train_data  = 255 * (train_data - train_min) / (train_max - train_min)
        val_data  = 255 * (val_data - train_min) / (train_max - train_min)
        test_data  = 255 * (test_data - train_min) / (train_max - train_min)

        train_data = train_data.astype(np.uint8)
        val_data = val_data.astype(np.uint8)
        test_data = test_data.astype(np.uint8)
        metadata = {'train_max': train_max, 'train_min': train_min}

        # Save the datasets to an HDF5 file
        with h5py.File(hdf5_file, "w") as hf:
            hf.create_dataset("train", data=train_data)
            hf.create_dataset("validation", data=val_data)
            hf.create_dataset("test", data=test_data)
            for key, value in metadata.items():
                hf.attrs[key] = value
        
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

    def plot_heat_map(self, idx):
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
        ax.imshow(self.field_list[idx])
        ax.set_xlabel("Longitudinal axis (m)")
        ax.set_ylabel("Transversal axis (m)")
        ax.set_title(f"Potential field heat map - Frame {idx}")
        plt.show()


def main():

    target_path = 'Tesis/datasets/highD/images_historic_1000ms/'
    dataset_location = 'Tesis/datasets/highD/groups_historic_1000ms/'
    if platform == 'darwin':
        dataset_location = '/Users/lmiguelmartinez/' + dataset_location
        target_path = '/Users/lmiguelmartinez/' + target_path
    else:
        dataset_location = "/home/lmmartinez/" + dataset_location
        target_path = "/home/lmmartinez/" + target_path

    if not os.path.exists(target_path):
        print("Creating target directory")
        os.makedirs(target_path, exist_ok=True)
    else:
        print("Target directory already exists - saving files to {}".format(target_path))

    rx = 50 # horizontal semiaxis of ellipse to consider as ROI
    ry = 6 # vertical semiaxis of ellipse to consider as ROI
    sx = 0.5
    sy = 0.1

    for dataset_index in range(1,61):
        print("-"*50)
        print("")
        print("Importing group list {} of 60".format(dataset_index))
        df_groups_list = pd.read_csv(dataset_location + str(dataset_index).zfill(2) + "_groups.csv")
        start = time()
        field = PotentialField(rx, ry, sx, sy, df_groups_list)
        field_list = field.calculate_field_list()
        end = time()
        print("Time taken is {}".format(end-start))
        print("The number of images generated is: {}".format(len(field_list)))
        filename = target_path + str(dataset_index).zfill(2) + '_images.hdf5'
        if os.path.exists(filename):
            try:
                os.remove(filename)
                print(f"File '{filename}' has been deleted.")
            except OSError as e:
                print(f"Error deleting the file '{filename}': {e}")
        # File does not exist, continue with the program
        print(f"File '{filename}' does not exist - saving to file.")
        field.save_to_hdf5(train_ratio=0.6, val_ratio=0.2, hdf5_file=filename)

if __name__ == "__main__":
    main()