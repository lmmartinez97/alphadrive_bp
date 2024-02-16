import glob
import numpy as np
import os
import sys

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(
        glob.glob(
            "/home/lmmartinez/CARLA/PythonAPI/carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass
import carla


import numpy as np
import carla

from .printers import print_blue, print_red

class PIDController:
    """
    Class for a generic PID controller.

    Args:
        params (dict): Dictionary containing PID parameters and timestep.
            Should contain keys 'Kp', 'Ki', 'Kd', and 'dt'.
    """
    def __init__(self, params):
        self.Kp = params.get('Kp', 1.0)
        self.Ki = params.get('Ki', 1.0)
        self.Kd = params.get('Kd', 1.0)
        self.dt = params.get('dt', 0.1)
        self.max_integral_threshold = params.get('max_integral_threshold', None)
        self.prev_error = 0
        self.integral = 0

    def run_step(self, error):
        """
        Runs one step of the PID controller.

        Returns:
            float: The control output.
        """
        self.integral += error * self.dt
        if self.max_integral_threshold and self.integral > self.max_integral_threshold:
            self.integral = 0 #reset integral to avoid windup
        derivative = (error - self.prev_error) / self.dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

class LongitudinalController(PIDController):
    """
    Class for longitudinal controller, extends PIDController.

    Args:
        params (dict): Dictionary containing longitudinal controller parameters.
            Inherits keys from PIDController's params and adds 'max_throttle' and 'max_brake'.
    """
    def __init__(self, params):
        super().__init__(params)
        self.max_throttle = params.get('max_throttle', 0.75)
        self.max_throttle_increment = params.get('max_throttle_increment', 0.1)
        self.max_brake = params.get('max_brake', 0.6)
        self.max_brake_increment = params.get('max_brake_increment', 0.1)
        self.error = 0
        self.prev_action = 0

    def saturation(self, value):
        """
        Applies saturation to the control output.

        Args:
            value (float): The control output.

        Returns:
            float: The saturated control output.
        """
        saturated_output = max(min(value, self.max_throttle), -self.max_brake)

        if saturated_output > 0:
            if saturated_output - self.prev_action > self.max_throttle_increment:
                saturated_output = self.prev_action + self.max_throttle_increment
            elif saturated_output - self.prev_action < -self.max_throttle_increment:
                saturated_output = self.prev_action - self.max_throttle_increment
        else:
            if saturated_output - self.prev_action > self.max_brake_increment:
                saturated_output = self.prev_action + self.max_brake_increment
            elif saturated_output - self.prev_action < -self.max_brake_increment:
                saturated_output = self.prev_action - self.max_brake_increment
        self.prev_action = saturated_output
        return saturated_output


    def run_step(self, target_speed, current_speed):
        """
        Runs one step of the longitudinal controller.

        Args:
            target_speed (float): The desired speed.
            current_speed (float): The current speed.

        Returns:
            float: Throttle or brake command.
        """
        self.target_speed = target_speed
        self.error = self.target_speed - current_speed
        return self.saturation(super().run_step(self.error))

class LateralController(PIDController):
    """
    Class for lateral controller, extends PIDController.

    Args:
        params (dict): Dictionary containing lateral controller parameters.
            Inherits keys from PIDController's params and adds 'max_steering'.
            
    Attributes:
        offset (float): The offset to be applied to the lateral control output.
    """
    def __init__(self, params):
        super().__init__(params)
        self.max_steering = params.get('max_steering', 1)
        self.max_steer_increment = params.get('max_steer_increment', 0.1)
        self.offset = 0
        self.error = 0
        self.prev_action = 0

    def saturation(self, value):
        """
        Applies saturation to the control output.

        Args:
            value (float): The control output.

        Returns:
            float: The saturated control output.
        """
        saturated_output = max(min(value, self.max_steering), -self.max_steering)
        if saturated_output - self.prev_action > self.max_steer_increment:
            saturated_output = self.prev_action + self.max_steer_increment
        elif saturated_output - self.prev_action < -self.max_steer_increment:
            saturated_output = self.prev_action - self.max_steer_increment
        self.prev_action = saturated_output
        return saturated_output
    
    def set_offset(self, offset):
        """Changes the offset"""
        self.offset = offset

    def run_step(self, target_location, current_location, current_yaw):
        """
        Runs one step of the lateral controller.

        Args:
            target_location (carla.Location): The target location.
            current_location (carla.Location): The current location.
            current_yaw (float): The current yaw angle.

        Returns:
            float: Steering command.
        """
        
        if self.offset != 0:
            # Displace the wp to the side
            r_vec = target_location.get_right_vector()
            target_location = target_location.location + carla.Location(x=self._offset*r_vec.x,
                                                         y=self._offset*r_vec.y)
        dx, dy = target_location.x - current_location.x, target_location.y - current_location.y
        self.target_location = target_location
        self.target_yaw = np.arctan2(dy, dx)
        self.error = self.target_yaw - current_yaw
        return self.saturation(super().run_step(self.error))

class CarController:
    """
    Class for controlling a car using longitudinal and lateral controllers.

    Args:
        longitudinal_params (dict): Parameters for longitudinal controller.
        lateral_params (dict): Parameters for lateral controller.
        reference (list): List of reference points for control.
        
    Longitudinal parameters:
        Kp (float): Proportional gain.
        Ki (float): Integral gain.
        Kd (float): Derivative gain.
        dt (float): Timestep.
        max_throttle (float): Maximum throttle command.
        max_brake (float): Maximum brake command.
        
    Lateral parameters:
        Kp (float): Proportional gain.
        Ki (float): Integral gain.
        Kd (float): Derivative gain.
        dt (float): Timestep.
        max_steering (float): Maximum steering command.
    
    Example:
        >>> longitudinal_params = {
        ...     'Kp': 1.0,
        ...     'Ki': 0.1,
        ...     'Kd': 0.5,
        ...     'dt': 0.03,
        ...     'max_throttle': 0.75,
        ...     'max_brake': 0.3
        ... }
        >>> lateral_params = {
        ...     'Kp': 1.0,
        ...     'Ki': 0.1,
        ...     'Kd': 0.5,
        ...     'dt': 0.03,
        ...     'max_steering': 0.8
        ... }
        >>> reference = [
        ...     (10, carla.Location(x=0, y=0)),
        ...     (10, carla.Location(x=10, y=0)),
        ...     (10, carla.Location(x=10, y=10))
        ... ]
        >>> controller = CarController(longitudinal_params, lateral_params, reference)
    """
    def __init__(self, longitudinal_params, lateral_params, reference):
        self.longitudinal_controller = LongitudinalController(longitudinal_params)
        self.lateral_controller = LateralController(lateral_params)
        self.reference = reference
        self.current_index = 0
        self.done = False

        print_blue("Initializated car controller")
        print_blue(f"Longitudinal params:")
        print_red(longitudinal_params)
        print_blue(f"Lateral params:")
        print_red(lateral_params)

    def update_reference(self, reference):
        """Updates the reference trajectory"""
        self.reference = reference
        self.current_index = 0
        self.done = False

    def run_step(self, current_speed, current_transform):
        """
        Executes a single step of the car controller, generating throttle, brake, and steering commands based on the current and target states of the car.

        Args:
            current_speed (float): The current speed of the car in km/h.
            current_transform (carla.Transform): The current transform (location and rotation) of the car.

        Returns:
            carla.Control: A carla.Control object containing the throttle, brake, and steering commands for the car.

        The function calculates the target speed and location based on the reference trajectory. It checks if the car has reached the target location, and if so, it updates the target to the next point in the reference trajectory. It then uses a longitudinal controller to generate a throttle or brake command based on the current and target speeds, and a lateral controller to generate a steering command based on the current and target locations and the car's current yaw angle. The throttle, brake, and steering commands are packaged into a carla.Control object and returned.
        """
        target_speed, target_location = self.reference[self.current_index]
        current_location = current_transform.location
        current_yaw = np.deg2rad(current_transform.rotation.yaw)

        # Check if the car has reached the target location
        if np.linalg.norm(np.array([target_location.x, target_location.y]) - np.array([current_location.x, current_location.y])) < 2.0:
            self.current_index += 1
            if self.current_index >= len(self.reference):
                self.current_index = len(self.reference) - 1
                self.done = True

        throttle_brake = self.longitudinal_controller.run_step(target_speed, current_speed)
        steering = self.lateral_controller.run_step(target_location, current_location, current_yaw)
        throttle = max(throttle_brake, 0)
        brake = max(-throttle_brake, 0)
        
        control = carla.VehicleControl(throttle=throttle, brake=brake, steer=steering)
        return control

