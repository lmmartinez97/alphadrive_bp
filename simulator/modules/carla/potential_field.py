import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygame

from matplotlib.backends.backend_agg import FigureCanvasAgg

class PotentialField:
    '''
    A class that represents the occupation information of a frame - See SceneData in dataviz.py.

    Attributes:
        radius_x, radius_y: semiaxis of the ellipse that is considered around the ego vehicle
        step_x, step_y: space discretization in each axis
        df_group_list: a list of vehicle grups as imported by the function read_groups of read_csv.py
    '''
    def __init__(self, radius_x: int = 50, radius_y: int = 6, step_x: float = 0.5, step_y: float = 0.1):
        self.rx, self.ry = radius_x, radius_y
        self.sx, self.sy = step_x, step_y
        self.dataframe = None
        self.num_x = int(2*self.rx / self.sx) + 1
        self.num_y = int(2*self.ry / self.sy) + 1
        self.grid = np.zeros([self.num_y, self.num_x], dtype = np.float32) #x dimension is rows, y dimension is columns
        self.x_pos, self.y_pos = np.linspace(-self.rx, self.rx, self.num_x), np.linspace(-self.ry, self.ry, self.num_y) #x and y interchanged so x is the horizontal dimension, and y is the vertical dimension
    
    def update(self, dataframe):
        '''
        Updates the dataframe from which the field is calculated
        Parameters:
                dataframe: dataframe that contains the information of the vehicles in the group.
            Returns:
                None
        '''
        self.grid = np.zeros([self.num_y, self.num_x], dtype = np.float32)
        self.dataframe = dataframe

    def calculate_field_ego(self, inner_group, ego_vehicle):
        '''
        Calculates the value of a potential field associated to an ego vehicle in the context of its group in a particular frame
        
        Parameters:
                inner_group: dataframe that contains the information of the vehicles in the group in a particular frame.
                ego_vehicle: pandas series that contains the information of the ego vehicle in a particular frame.
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
            k = np.linalg.norm((vehicle.xVelocity, vehicle.yVelocity))
            grid = get_field_value(dx, dy, a, b, c, k)
            return grid

        grid = np.zeros([self.num_y, self.num_x], dtype = np.float32) #x dimension is rows, y dimension is columns
        for _, row in inner_group.iterrows():
            grid = np.add(grid, calculate_parameters_and_field_value(ego_vehicle=ego_vehicle, vehicle=row))

        return grid
    
    def calculate_field(self):
        '''
        Performs potential field calculations for every group in df_group_list
        
        Parameters:
                None
            Returns:
                An np.array in which each element is the potential field representation of the associated group.
        '''
        self.field_list = None
        self.dataframe["frame"] = - self.dataframe["frame"] + self.dataframe["frame"].max() + 1 # determine relative frame step to atenuate distant values
        #Current frame will have a value of 1, previous frame will have a value of 2, nth frame will have a value of n
        ego_vehicle_id = self.dataframe[self.dataframe["hero"] == 1]["id"].iloc[0]
        temp_grid = np.zeros([self.num_y, self.num_x], dtype = np.float32) #x dimension is rows, y dimension is columns
        for frame_number, inner_group in self.dataframe.groupby("frame"): #add the field for every frame
            frame_attenuation = np.exp(-0.65*(inner_group.iloc[0].frame -1))
            ego_vehicle = inner_group[inner_group["id"] == ego_vehicle_id].iloc[0]
            self.field_list = np.add(self.calculate_field_ego(inner_group, ego_vehicle) * frame_attenuation, temp_grid)
        self.field_list = np.asarray(self.field_list)

        return self.field_list.reshape(-1, self.num_y, self.num_x, 1)
        
    def plot_field(self, idx = None):
        '''
        Plots a 3d graph and a heat map of the potential field of a given group in the variable idx.
        
        Parameters:
                None
            Returns:
                None
        '''
        fig = plt.figure(figsize=(15, 8))

        #plot 3d field
        ax3d = fig.add_subplot(1, 2, 1, projection='3d')
        X, Y = np.meshgrid(self.x_pos, self.y_pos)
        surf = ax3d.plot_surface(X, Y, self.field_list, cmap = "viridis")
        ax3d.set_xlabel("Longitudinal axis (m)")
        ax3d.set_ylabel("Transversal axis (m)")
        ax3d.set_zlabel("Potential field magnitude (-)")
        ax3d.set_title("3d plot - Normalized between 0 and 1")
        fig.colorbar(surf, orientation = 'horizontal', pad = 0.2)

        #plot heatmap
        ax2d = fig.add_subplot(1, 2, 2)
        img = ax2d.imshow(self.field_list)
        ax2d.set_xlabel("Longitudinal axis (px)")
        ax2d.set_ylabel("Transversal axis (px)")
        ax2d.set_title("Potential field heat map")
        fig.colorbar(img, orientation = 'horizontal', pad = 0.2)

        fig.suptitle(f"Potential field representation - Frame {idx}")
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
        y_index = int((self.field_list.shape[0] - 1) / 2)
        x = self.x_pos
        y = self.field_list[y_index]
        ax.plot(x, y)
        ax.set_xlabel("Transversal axis (m)")
        ax.set_ylabel("Potential field magnitude (-)")
        ax.set_title(f"Ego longitudinal x axis - Frame {idx}")

        return fig

    def plot_heat_map(self, idx = None):
        '''
        Plots the heat map of the potential field of a given group in the variable idx.
        
        Parameters:
                idx: index of the group that will be represented
            Returns:
                None
        '''
        idx = 1 if idx is None else idx
        fig, ax = plt.subplots()
        ax.imshow(self.field_list)
        ax.set_xlabel("Longitudinal axis (m)")
        ax.set_ylabel("Transversal axis (m)")
        ax.set_title(f"Potential field heat map - Frame {idx}")

        return fig
    
    def figure_to_surface(matplotlib_figure):
        """
        Converts a Matplotlib figure to a Pygame surface.

        Parameters:
        - matplotlib_figure (matplotlib.figure.Figure): The Matplotlib figure to convert.

        Returns:
        - pygame.Surface: The Pygame surface containing the rendered Matplotlib figure.

        Examples:
        >>> # Example 1:
        >>> fig, ax = plt.subplots()
        >>> x = np.linspace(0, 2 * np.pi, 100)
        >>> y = np.sin(x)
        >>> ax.plot(x, y)
        >>> ax.set_title("Sin Wave")
        >>> pygame_surface = figure_to_surface(fig)

        >>> # Example 2:
        >>> fig, ax = plt.subplots()
        >>> data = np.random.rand(10, 10)
        >>> ax.imshow(data, cmap='viridis')
        >>> ax.set_title("Random Image")
        >>> pygame_surface = figure_to_surface(fig)
        """
        # Create a FigureCanvasAgg to render the figure
        canvas = FigureCanvasAgg(matplotlib_figure)
        canvas.draw()

        # Extract the pixel buffer from the rendered figure
        w, h = canvas.get_width_height()
        buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape((h, w, 3))

        # Create a Pygame surface and copy the pixel buffer onto it
        pygame_surface = pygame.surfarray.make_surface(buf)

        return pygame_surface    

    def render(self, display, x_pos, y_pos):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (x_pos, y_pos))