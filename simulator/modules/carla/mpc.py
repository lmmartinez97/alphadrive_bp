"""MPC MODULE"""
import numpy as np

from copy import deepcopy
from rich import print
from scipy.optimize import minimize
from scipy.spatial import KDTree
from typing import Any, Dict, List, Tuple

from .printers import print_blue, print_green


class MPCController:
    """Model Predictive Controller class for vehicle control."""

    def __init__(self, parameters: Dict = None):
        """
        Initializes the MPC (Model Predictive Control) class.

        Args:
            parameters (Dict): A dictionary containing the parameters for the MPC model.
                The dictionary should have the following keys:
                - mass (float): Mass of the vehicle (kg)
                - L (float): Wheelbase of the vehicle (meters)
                - a (float): Distance from front axle to center of mass (meters)
                - b (float): Distance from rear axle to center of mass (meters)
                - frontal_area (float): Frontal area of the vehicle (m^2)
                - drag_coefficient (float): Aerodynamic drag coefficient
                - max_acceleration (float): Maximum achievable acceleration (m/s^2)
                - max_deceleration (float): Maximum achievable deceleration (m/s^2)
                - air_density (float): Density of air (kg/m^3)
                - gravity (float): Acceleration due to gravity (m/s^2)
                - dt (float): Time step for integration (seconds)
                - prediction_horizon (int): Prediction horizon (number of time steps)
                - max_steering_rate (float): Maximum rate of change of steering angle per time step
            reference_trajectory (list): A list of [[x, y, z], velocity] points representing the reference trajectory.
            offset (float): The offset to be added to the reference trajectory.

        Example:
        parameters = {
            'mass': 1500,
            'L': 2.7, #wheelbase
            'a': 1.2, #distance from CoG to front axle
            'b': 1.5, #distance from CoG to rear axle
            'frontal_area': 2.4,
            'drag_coefficient': 0.24,
            'max_acceleration': 2.5,
            'max_deceleration': 4,
            'air_density': 1.2,
            'gravity': 9.81,
            'dt': self.simulation_period,
            'prediction_horizon': 4,
            'max_steering_rate': 0.1,
            'tracking_cost_weight': 1,
            'velocity_cost_weight': 1,
            'yaw_cost_weight': 1,
            'steering_rate_cost_weight': 1,
            'pedal_rate_cost_weight': 1,
            'exponential_decay_rate': 0.6,
        }
            mpc = MPC(parameters, reference, offset)
        """
        # Define constant parameters for the model
        # Geometric parameters
        self.mass = parameters.get('mass', 1500)  # Mass of the vehicle (kg)
        self.L = parameters.get('L', 2.7)  # Wheelbase of the vehicle (meters)
        self.a = parameters.get('a', 1.2)  # Distance from front axle to center of mass (meters)
        self.b = parameters.get('b', 1.5)  # Distance from rear axle to center of mass (meters)
        self.frontal_area = parameters.get('frontal_area', 2.4)  # Frontal area of the vehicle (m^2)
        self.drag_coefficient = parameters.get('drag_coefficient', 0.24)  # Aerodynamic drag coefficient

        # Dynamic parameters
        self.max_acceleration = parameters.get('max_acceleration', 2.5)  # Maximum achievable acceleration (m/s^2)
        self.max_deceleration = parameters.get('max_deceleration', 4)  # Maximum achievable deceleration (m/s^2)
        
        # Environmental parameters
        self.air_density = parameters.get('air_density', 1.2)  # Density of air (kg/m^3)
        self.gravity = parameters.get('gravity', 9.81)  # Acceleration due to gravity (m/s^2)

        # Control parameters
        self.dt = parameters.get('dt', 0.1)  # Time step for integration (seconds)
        self.prediction_horizon = parameters.get('prediction_horizon', 10)  # Prediction horizon (number of time steps)
        self.max_steering_rate = parameters.get('max_steering_rate', 0.05)  # Maximum rate of change of steering angle per time step
        self.max_pedal_rate = parameters.get('max_pedal_rate', 0.1)  # Maximum rate of change of pedal position per time step
        
        # Cost function parameters
        self.tracking_cost_weight = parameters.get('tracking_cost_weight', 1.0)  # Weight for tracking cost
        self.velocity_cost_weight = parameters.get('velocity_cost_weight', 1.0)  # Weight for velocity cost
        self.yaw_cost_weight = parameters.get('yaw_cost_weight', 1.0)  # Weight for yaw cost
        self.steering_rate_cost = parameters.get('steering_rate_cost_weight', 1.0)  # Weight for steering rate cost
        self.pedal_rate_cost = parameters.get('pedal_rate_cost_weight', 1.0)  # Weight for pedal rate cost
        self.exponential_decay = parameters.get('exponential_decay_rate', None) #decay factor for the cost function

        self.reference = []  # Reference trajectory
        self.prev_control = [0, 0]  # Previous control inputs steering, pedal - initalized for optimization process
        
        self.offset = 0.0        
        self.last_tracked_index = 0  # Index of the last tracked point in the reference trajectory

        #Define optimization bounds
        self.bounds_steer = (-0.75, 0.75)
        self.bounds_pedal = (-0.5, 0.75)
        self.bounds = [self.bounds_steer, self.bounds_pedal] * self.prediction_horizon
        
        #Define optimization method
        self.method = 'SLSQP'
        self.method_opts = {'dict_config':{'maxiter': 1000, 'disp': False, 'ftol': 1e-3, 'eps': np.deg2rad(.5)}, 
                            'jacobian_type': '2-point'}
        
        np.set_printoptions(precision=6, suppress=True)
        
        self.steer_conversion = 1.22 #a value of 1 in carla.VehicleControl equates to 1.22 radians at the wheels
        
        print_blue("MPC controller initialized with params:")
        [print_green(f"{k}: {v}") for k, v in parameters.items()]
        
        
        
    def __call__(self, state: Dict) -> Any:
        """
        Optimize control inputs using model predictive control.

        Args:
            current_state (dict): Current state of the vehicle, containing components [x, y, yaw, velocity].

        Returns:
            dict: Optimized control inputs for the vehicle, containing [steering, throttle, brake].
        """
        return self.optimize_control(current_state=state)
    
    def find_closest_index(self, current_state):
        
        # Find the closest point in the reference trajectory
        _, closest_index = self.reference_tree.query([current_state['x'], current_state['y']])
        #Ensure that the closest index is positioned ahead of the vehicle
        #calculate the orientation of the vector that goes from the vehicle to the closest point in the vehicle frame
        dx = self.reference[closest_index]['x'] - current_state['x']
        dy = self.reference[closest_index]['y'] - current_state['y']
        angle = np.arctan2(dy, dx)
        #calculate the difference between the orientation of the vehicle and the orientation of the vector
        diff_angle = np.abs(current_state['yaw'] - angle)
        #if the difference is greater than 90 degrees, then the closest point is behind the vehicle
        if diff_angle > np.pi/2:
            closest_index += 1

        return closest_index

    def model(self, state: Dict, control: Dict) -> Dict:
        """
        Model function for the dynamic bicycle model.

        Args:
            state (dict): Current state of the vehicle, containing components [x, y, yaw, velocity, pitch].
            control (dict): Control inputs for the vehicle, containing [steering, throttle, brake].

        Returns:
            dict: Derivatives of state variables, containing [x_dot, y_dot, yaw_rate, velocity_dot].
        """
        # Calculate net longitudinal force
        steer_angle = control['steering'] * self.steer_conversion
        throttle_pressure = control['pedal'] if control['pedal'] > 0 else 0
        brake_pressure = control['pedal'] if control['pedal'] < 0 else 0
        net_force = (self.max_acceleration * throttle_pressure + self.max_deceleration * brake_pressure)*self.mass
        
        # Calculate aerodynamic drag force
        velocity = state['velocity']
        drag_force = - 0.5 * self.drag_coefficient * self.air_density * self.frontal_area * np.square(velocity)
        
        # Calculate gravitational force component parallel to the road surface
        road_angle = state['pitch']  # Using pitch angle as road inclination
        weight_force = -self.gravity * np.sin(road_angle) * self.mass  # Incorporate vehicle mass here
            
        # Calculate longitudinal acceleration
        acceleration = (net_force + drag_force + weight_force) / self.mass  # Divide by mass to get acceleration
        
        # Calculate slip angle at the center of gravity
        yaw_rate = state['velocity'] * np.tan(steer_angle) / self.L
        
        # Calculate derivatives of position
        x_dot = state['velocity'] * np.cos(state['yaw'])
        y_dot = state['velocity'] * np.sin(state['yaw'])
        
        return {'x_dot': x_dot, 'y_dot': y_dot, 'yaw_rate': yaw_rate, 'velocity_dot': acceleration}
    
    def calculate_velocity_cost(self, closest_index: int, state_trajectory: Dict, exponential_decay: float = 1.0) -> float:
        """
        Calculate the velocity cost for the predicted trajectory.
        
        If exponential decay is true, a decay factor is applied to mitigate the important of distant points in the trajectory.
        """
        cost = 0
        for i, state in enumerate(state_trajectory):
            if closest_index + i < len(self.reference):
                ref_state = self.reference[closest_index + i]
                cost += np.linalg.norm([state['velocity'] - ref_state['velocity']]) * exponential_decay**i
        return cost
    
    def calculate_yaw_cost(self, closest_index: int, state_trajectory: Dict, exponential_decay: float = 1.0) -> float:
        """
        Calculate the yaw cost for the predicted trajectory.
        
        If exponential decay is true, a decay factor is applied to mitigate the important of distant points in the trajectory.
        """
        
        cost = 0
        for i, state in enumerate(state_trajectory):
            if closest_index + i < len(self.reference):
                ref_state = self.reference[closest_index + i]
                angle_error = np.abs(state['yaw'] - ref_state['yaw'])
                cost += angle_error * exponential_decay**i
                
        return cost
    
    def calculate_tracking_cost(self, closest_index: int, state_trajectory: Dict, exponential_decay: float = 1.0, verbose: bool = False) -> float:
        """
        Calculate the tracking cost for the predicted trajectory.
        
        If exponential decay is true, a decay factor is applied to mitigate the important of distant points in the trajectory.
        """

        cost = 0
        for i, state in enumerate(state_trajectory):
            if closest_index + i < len(self.reference)-1:
                ref_state = self.reference[closest_index + i]
                next_ref_state = self.reference[closest_index + i + 1]
            else: #if we are at the end of the reference trajectory, use the last two reference points
                ref_state = self.reference[-2]
                next_ref_state = self.reference[-1]
            #calculate distance between the vehicle and the line that goes through the two reference points
            p1 = np.array([ref_state['x'], ref_state['y']])
            p2 = np.array([next_ref_state['x'], next_ref_state['y']])
            p3 = np.array([state['x'], state['y']])
            distance = np.abs(np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1))
            if verbose:
                print(f"Distance to line {i}: {distance}")
            cost += distance * exponential_decay**i
        return cost
    
    def calculate_control_rate_cost(self, control_sequence: List) -> float:
        """
        Calculate the control rate cost for the predicted trajectory.
        """
        cost = 0
        for i in range(len(control_sequence) - 1):
            cost += np.abs(control_sequence[i] - control_sequence[i + 1])
        return cost
    
    def calculate_state_trajectory(self, current_state: Dict, control_sequence: List) -> Dict:
        # Calculate state trajectory using model and control inputs
        state_trajectory = [current_state]
        for i in range(self.prediction_horizon):
            control = {'steering': control_sequence[i, 0], 'pedal': control_sequence[i, 1]}
            derivatives = self.model(state_trajectory[-1], control)
            next_state = self.calculate_next_state(state_trajectory[-1], derivatives, self.dt)
            state_trajectory.append(next_state)
            
        return state_trajectory
    
    def calculate_total_cost(self, control_sequence, verbose = False) -> float:
        """
        Calculate the total cost for the predicted trajectory.
        
        If exponential decay is true, a decay factor is applied to mitigate the important of distant points in the trajectory.
        """
        control_sequence = np.reshape(control_sequence, (-1, 2))
        state_trajectory = self.calculate_state_trajectory(self.current_state, control_sequence=control_sequence)
        tracking_cost = self.calculate_tracking_cost(self.closest_index, state_trajectory, self.exponential_decay, verbose=False)
        velocity_cost = self.calculate_velocity_cost(self.closest_index, state_trajectory, self.exponential_decay)
        yaw_cost = self.calculate_yaw_cost(self.closest_index, state_trajectory, self.exponential_decay)
        steering_rate_cost = self.calculate_control_rate_cost(control_sequence[:, 0])
        pedal_rate_cost = self.calculate_control_rate_cost(control_sequence[:, 1])
        cost = self.tracking_cost_weight * tracking_cost + self.velocity_cost_weight * velocity_cost + self.yaw_cost_weight * yaw_cost + self.steering_rate_cost * steering_rate_cost + self.pedal_rate_cost * pedal_rate_cost
        if verbose:
            print("Tracking cost: ", tracking_cost)
            print("Velocity cost: ", velocity_cost)
            print("Yaw cost: ", yaw_cost)
            print("Steering rate cost: ", steering_rate_cost)
            print("Pedal rate cost: ", pedal_rate_cost)
            print("Total cost: ", cost)
        return cost

    def optimize_control(self, current_state: Dict) -> Tuple[Dict, List[Dict]]:
        """
        Optimize the control inputs for the vehicle over the prediction horizon.

        This function uses the current state of the vehicle and a cost function to optimize the control inputs
        (steering, throttle, and brake) over the prediction horizon. The cost function is designed to minimize
        the deviation from the reference trajectory and the deviation in velocity. The optimization is performed
        using the scipy.optimize.minimize function, which finds the control inputs that minimize the cost function
        subject to the constraints defined in the function.

        The function returns the first set of optimized control inputs, which can be used to control the vehicle
        in the next time step. The rest of the optimized control inputs can be used to reconstruct the predicted
        trajectory for plotting.

        Args:
            current_state (dict): The current state of the vehicle. The state is a dictionary that includes the
                                following keys: 'x', 'y', 'yaw', and 'velocity'.

        Returns:
            dict: The optimized control inputs for the next time step. The dictionary includes the following keys:
                'steering', 'pedal'.
            reconstructed states: list of states reconstructed using the model and control inputs
        """
        self.closest_index = self.find_closest_index(current_state)
        self.last_tracked_index = self.closest_index
        initial_guess = np.tile(self.prev_control, (self.prediction_horizon, 1)).flatten()
        self.current_state = current_state
        
        # Define constraints for the optimization
        self.constraints = []
        for i in range(self.prediction_horizon):
            if i == 0:
                self.constraints.extend([
                    {'type': 'ineq', 'fun': lambda x: self.max_steering_rate - np.abs(x[i*2] - self.prev_control[0])},  # absolute steering change rate <= max_steering_rate
                    {'type': 'ineq', 'fun': lambda x: self.max_pedal_rate - np.abs(x[i*2 + 1] - self.prev_control[1])},  # absolute pedal change rate <= max_pedal_rate
                ])
            else:
                self.constraints.extend([
                    {'type': 'ineq', 'fun': lambda x: self.max_steering_rate - np.abs(x[i*2] - x[(i-1)*2])},  # absolute steering change rate <= max_steering_rate
                    {'type': 'ineq', 'fun': lambda x: self.max_pedal_rate - np.abs(x[i*2 + 1] - x[(i-1)*2 + 1])},  # absolute pedal change rate <= max_pedal_rate
                ])

        # Optimize control inputs
        result = minimize(self.calculate_total_cost, 
                          x0=initial_guess, 
                          constraints=self.constraints, 
                          bounds=self.bounds,
                          method='SLSQP',
                          options=self.method_opts['dict_config'],
                          jac=self.method_opts['jacobian_type'])

        # Reshape the result into a 2D array - [steer, pedal]
        optimized_control_sequence = np.reshape(result.x, (-1, 2))
        self.calculate_total_cost(optimized_control_sequence, verbose=False)
        reconstructed_states = self.calculate_state_trajectory(current_state, optimized_control_sequence)
        
        #if the predicted velocity of the next iteration is zero, control was unsuccessful, so we go straigt with throttle = 0.3
        if reconstructed_states[0]['velocity'] < 1:
            print("Control failed, going straight")
            return_dict = {'steering': 0,
                            'throttle': 0.3,
                            'brake': 0}

        else:
            print_green("Optimized control sequence: ", optimized_control_sequence[0])
            return_dict = {'steering': optimized_control_sequence[0, 0],
                        'throttle': optimized_control_sequence[0, 1] if optimized_control_sequence[0, 1] > 0 else 0,
                        'brake': -optimized_control_sequence[0, 1] if optimized_control_sequence[0, 1] < 0 else 0}         
        
        self.prev_control = optimized_control_sequence[0]
        
        return return_dict, reconstructed_states

    def calculate_next_state(self, current_state, derivatives, dt):
        """
        Calculate the next state of the vehicle using Euler integration.

        Args:
            current_state (dict): Current state of the vehicle, containing components [x, y, yaw, velocity].
            derivatives (dict): Derivatives of state variables, containing [x_dot, y_dot, yaw_rate, velocity_dot].
            dt (float): Time step for integration.

        Returns:
            dict: Next state of the vehicle after integrating for the given time step.
        """
        next_state = {
            'x': current_state['x'] + derivatives['x_dot'] * dt,
            'y': current_state['y'] + derivatives['y_dot'] * dt,
            'yaw': current_state['yaw'] + derivatives['yaw_rate'] * dt,
            'velocity': current_state['velocity'] + derivatives['velocity_dot'] * dt,
            'pitch': current_state['pitch'], # pitch remains constant
        }
        #normalize yaw
        next_state['yaw'] = (next_state['yaw'] + np.pi) % (2 * np.pi) - np.pi
        
        return next_state

    def update_reference(self, reference_trajectory):
        """
        Update the reference trajectory for the MPC controller.

        Args:
            reference_trajectory (list): List of [[float, float, float], float], representing locations and velocities.
                                         velocities are in km/h, and locations attributes are in meters.
                                         
        The controller expects the reference to be a list of dictionaries, each containing the following keys:
        - x (float): x-coordinate of the reference point (meters)
        - y (float): y-coordinate of the reference point (meters)
        - yaw (float): Yaw angle at the reference point (radians)
        - velocity (float): Velocity at the reference point (m/s)

        """
        self.reference = []
        for i in range(len(reference_trajectory)):
            location, velocity = reference_trajectory[i]
            if i < len(reference_trajectory) - 1:
                next_location, _ = reference_trajectory[i + 1]
                dx = next_location[0] - location[0]
                dy = next_location[1] - location[1]
                yaw = np.arctan2(dy, dx)
            self.reference.append({'x': location[0], 'y': location[1], 'yaw': yaw, 'velocity': velocity / 3.6})

        self.ref_copy = deepcopy(self.reference)
        self.reference_tree = KDTree(np.array([[ref['x'], ref['y']] for ref in self.reference]))

    def set_offset(self, offset):
        """
        Set the offset for the reference trajectory.

        To add the offset to the reference trajectory, a copy of the original (default) trajectory is used.
        This method loops over the remaining points in the original reference, and calculates the new x and y. It then updates the
        reference that is being used by the controller, as well as the KDTree that is used for finding the closest point to the current state.
        Args:
            offset (float): The offset to be added to the reference trajectory.
        """
        self.offset = offset
        for idx, ref in enumerate(self.ref_copy[self.last_tracked_index:]):
            # Calculate the offset in the x and y directions based on the yaw angle of the original trajectory
            dx = offset * np.cos(ref['yaw'] + np.pi / 2)
            dy = offset * np.sin(ref['yaw'] + np.pi / 2)

            # Add the offset to the x and y coordinates of the original reference trajectory
            # This way the offset is applied on the original trajectory, not on the previous offset
            self.reference[self.last_tracked_index + idx]['x'] = ref['x'] + dx
            self.reference[self.last_tracked_index + idx]['y'] = ref['y'] + dy
        
        #update reference tree with information of the new reference
        self.reference_tree = KDTree(np.array([[ref['x'], ref['y']] for ref in self.reference]))

            
    def calculate_errors(self, current_state, verbose=False):
        """
        Calculate the cross track error and yaw.
        
        Args:
            current_state (dict): Current state of the vehicle, containing components [x, y, yaw, velocity].
            verbose (bool): If True, print the calculated errors.
        
        Returns:
            dict: References and errors for the controller to use.
        """
        #Calculate lateral error
        closest_index = self.find_closest_index(current_state)
        ref = self.reference[closest_index]
        dx = current_state['x'] - ref['x']
        dy = current_state['y'] - ref['y']
        lateral_error = np.sqrt(dx**2 + dy**2)*np.sign(dy)*np.cos(ref['yaw'] - np.arctan2(dy, dx))
        
        #calculate velocity error
        velocity_error = current_state['velocity'] - ref['velocity']
        
        #calculate yaw error
        yaw_error = current_state['yaw'] - ref['yaw']
        if verbose:
            print("Lateral Error: ", lateral_error)
            print("Velocity Error: ", velocity_error)
            print("Yaw Error: ", yaw_error)
            
        return {'pos_target': [ref['x'], ref['y']],
                'velocity_target': ref['velocity'],
                'yaw_target': ref['yaw'],
                'lateral_error': lateral_error,
                'velocity_error': velocity_error,
                'yaw_error': yaw_error}
    
    def is_done(self):
        """
        Check if the controller has reached the end of the reference trajectory.

        Returns:
            bool: True if the controller has reached the end of the reference trajectory, False otherwise.
        """
        ret = False
        if self.last_tracked_index >= len(self.reference) - 1:
            self.last_tracked_index = 0
            ret = True
        return ret

