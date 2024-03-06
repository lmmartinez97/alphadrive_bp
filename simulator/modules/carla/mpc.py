from typing import Any, Dict
import numpy as np
from scipy.optimize import minimize

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

        Example:
            parameters = {
                'mass': 1500,
                'L': 2.7,
                'a': 1.2,
                'b': 1.5,
                'frontal_area': 2.4,
                'drag_coefficient': 0.24,
                'max_acceleration': 5.0,
                'max_deceleration': 3.0,
                'air_density': 1.2,
                'gravity': 9.81,
                'dt': 0.1,
                'prediction_horizon': 10,
                'max_steering_rate': 0.05,
                'tracking_cost_weight': 1.0,
                'velocity_cost_weight': 1.0
            }
            mpc = MPC(parameters)
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
        self.max_acceleration = parameters.get('max_acceleration', 5.0)  # Maximum achievable acceleration (m/s^2)
        self.max_deceleration = parameters.get('max_deceleration', 3.0)  # Maximum achievable deceleration (m/s^2)
        
        # Environmental parameters
        self.air_density = parameters.get('air_density', 1.2)  # Density of air (kg/m^3)
        self.gravity = parameters.get('gravity', 9.81)  # Acceleration due to gravity (m/s^2)

        # Control parameters
        self.dt = parameters.get('dt', 0.1)  # Time step for integration (seconds)
        self.prediction_horizon = parameters.get('prediction_horizon', 10)  # Prediction horizon (number of time steps)
        self.max_steering_rate = parameters.get('max_steering_rate', 0.05)  # Maximum rate of change of steering angle per time step
        
        # Cost function parameters
        self.tracking_cost_weight = parameters.get('tracking_cost_weight', 1.0)  # Weight for tracking cost
        self.velocity_cost_weight = parameters.get('velocity_cost_weight', 1.0)  # Weight for velocity cost
        
        self.reference = None  # Reference trajectory
        self.prev_control = [0, 0, 0]  # Previous control inputs steering, throttle, brake - initalized for optimization process

    def __call__(self, state: Dict) -> Any:
        """
        Optimize control inputs using model predictive control.

        Args:
            current_state (dict): Current state of the vehicle, containing components [x, y, yaw, velocity].

        Returns:
            dict: Optimized control inputs for the vehicle, containing [steering, throttle, brake].
        """
        return self.optimize_control(current_state=state)

    def model(self, state, control):
        """
        Model function for the dynamic bicycle model.

        Args:
            state (dict): Current state of the vehicle, containing components [x, y, yaw, velocity, pitch].
            control (dict): Control inputs for the vehicle, containing [steering, throttle, brake].

        Returns:
            dict: Derivatives of state variables, containing [x_dot, y_dot, yaw_rate, velocity_dot].
        """
        # Calculate net longitudinal force
        throttle_pressure = control['throttle']
        brake_pressure = control['brake']
        net_force = (self.max_acceleration * throttle_pressure - self.max_deceleration * brake_pressure)*self.mass
        
        # Calculate aerodynamic drag force
        velocity = state['velocity']
        drag_force = 0.5 * self.drag_coefficient * self.air_density * self.frontal_area * state['velocity']**2
        
        # Calculate gravitational force component parallel to the road surface
        road_angle = state['pitch']  # Using pitch angle as road inclination
        weight_force = -self.gravity * np.sin(road_angle) * self.mass  # Incorporate vehicle mass here
            
        # Calculate longitudinal acceleration
        acceleration = (net_force - drag_force + weight_force) / self.mass  # Divide by mass to get acceleration
        
        # Calculate yaw rate
        yaw_rate = state['velocity'] * np.tan(control['steering']) / self.L
        
        # Calculate derivatives of position
        x_dot = state['velocity'] * np.cos(state['yaw'])
        y_dot = state['velocity'] * np.sin(state['yaw'])
        
        return {'x_dot': x_dot, 'y_dot': y_dot, 'yaw_rate': yaw_rate, 'velocity_dot': acceleration}


    def optimize_control(self, current_state):
        """
        Optimize control inputs using model predictive control.

        Args:
            current_state (dict): Current state of the vehicle, containing components [x, y, yaw, velocity].

        Returns:
            dict: Optimized control inputs for the vehicle, containing [steering, throttle, brake].
        """
        # Find the index of the closest point in the reference trajectory
        closest_index = self.find_closest_index(current_state)

        # Update last tracked index for faster search next time
        self.last_tracked_index = closest_index

        def cost_function(control):
            # Calculate state trajectory using model and control inputs
            state_trajectory = [current_state]
            for _ in range(self.prediction_horizon):
                derivatives = self.model(state_trajectory[-1], {'steering': control[0], 'throttle': control[1], 'brake': control[2]})
                next_state = self.calculate_next_state(state_trajectory[-1], derivatives, self.dt)
                state_trajectory.append(next_state)
            
            # Calculate cost based on deviation from reference trajectory
            cost = 0
            tracking_cost = 0
            for i, state in enumerate(state_trajectory):
                if closest_index + i < len(self.reference):
                    # Calculate Euclidean distance between predicted and reference positions
                    ref_state = self.reference[closest_index + i]
                    tracking_cost += np.linalg.norm([state['x'] - ref_state['x'], state['y'] - ref_state['y']])
            cost += self.tracking_cost_weight * tracking_cost
            # Add cost for velocity deviation
            cost += self.velocity_cost_weight * np.linalg.norm([state['velocity'] - ref_state['velocity']])

            return cost

        # Initial guess for control inputs

        # Define optimization constraints
        constraints = ({'type': 'ineq', 'fun': lambda x: x[1] - 1},  # throttle <= 1
                       {'type': 'ineq', 'fun': lambda x: x[2] - 1},  # brake <= 1
                       {'type': 'ineq', 'fun': lambda x: x[0] - 1},  # steering <= 1
                       {'type': 'ineq', 'fun': lambda x: -x[0] - 1},  # steering >= -1
                       {'type': 'ineq', 'fun': lambda x: x[0] - (self.prev_control[0] + self.max_steering_rate)},  # steering change rate <= 0.05
                       {'type': 'ineq', 'fun': lambda x: (self.prev_control[0] - self.max_steering_rate) - x[0]})# steering change rate >= -0.05

        # Perform optimization - control in format [steering, throttle, brake]
        result = minimize(cost_function, self.prev_control, constraints=constraints)

        # Extract optimized control inputs
        optimized_control = {'steering': result.x[0], 'throttle': result.x[1], 'brake': result.x[2]}

        # Update previous steering angle
        self.prev_control = [result.x[0], result.x[1], result.x[2]]

        return optimized_control

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
            'velocity': current_state['velocity'] + derivatives['velocity_dot'] * dt
        }
        return next_state

    def update_reference(self, reference_trajectory):
        """
        Update the reference trajectory for the MPC controller.

        Args:
            reference_trajectory (list): List of reference states for the vehicle control.

        """
        self.reference = reference_trajectory
