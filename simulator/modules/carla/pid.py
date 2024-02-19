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


from .printers import print_blue, print_red

from rich import print

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
        self.error = 0
        print(self.__dict__)

    def run_step(self, error):
        """
        Runs one step of the PID controller.

        Returns:
            float: The control output.
        """
        self.error = error
        if self.max_integral_threshold and self.integral > self.max_integral_threshold:
            self.integral = 0 #reset integral to avoid windup
        self.integral += error * self.dt
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
        saturated_output = np.clip(value, -self.max_brake, self.max_throttle)
        increment = self.max_throttle_increment if saturated_output > 0 else self.max_brake_increment
        saturated_output = np.clip(saturated_output, self.prev_action - increment, self.prev_action + increment)

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
        self.action = self.saturation(super().run_step(self.error))
        # print("Longitudinal controller action breakdown")
        # print("Integral: ", self.Ki * self.integral)
        # print("Proportional: ", self.Kp * self.error)
        # print("Derivative: ", self.Kd * ((self.error - self.prev_error) / self.dt))
        # print("Action: ", self.action)
        # print()
        return self.action


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
        self.max_steer_increment = params.get('max_steer_increment', 1)
        self.distance_threshold = params.get('distance_threshold', 2)
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
        saturated_output = np.clip(value, -self.max_steering, self.max_steering)
        saturated_output = np.clip(saturated_output, self.prev_action - self.max_steer_increment, self.prev_action + self.max_steer_increment)
        self.prev_action = saturated_output
        return saturated_output

    def run_step(self, target_location, current_transform):
        """
        Runs one step of the lateral controller.

        Args:
            target_location (carla.Location): The current target location.
            current_transform (carla.Transform): The current transform of the ego vehicle.

        Returns:
            float: Steering command.
        """
        self.target_location = target_location
        current_location = current_transform.location
        dx, dy = target_location.x - current_location.x, target_location.y - current_location.y
        distance_to_target = np.hypot(dx, dy)

        # If the distance to the target is below a certain threshold, maintain value of error
        if distance_to_target > self.distance_threshold:
            v_vec = current_transform.get_forward_vector()
            v_vec = np.array([v_vec.x, v_vec.y, 0.0])
            w_vec = np.array([self.target_location.x - current_location.x,
                            self.target_location.y - current_location.y,
                            0.0])
            wv_linalg = np.linalg.norm(w_vec) * np.linalg.norm(v_vec)
            if wv_linalg == 0:
                self.error = 0
            else:
                self.error = np.arccos(np.clip(np.dot(w_vec, v_vec) / (wv_linalg), -1.0, 1.0))
            _cross = np.cross(v_vec, w_vec)
            if _cross[2] < 0:
                self.error *= -1.0
        else:
            self.error = 0
                        
        self.action = self.saturation(super().run_step(self.error))
        # print("Lateral controller action breakdown")
        # print("Integral: ", self.Ki * self.integral)
        # print("Proportional: ", self.Kp * self.error)
        # print("Derivative: ", self.Kd * ((self.error - self.prev_error) / self.dt))
        # print("Action: ", self.action)
        # print()
        return self.action

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
        self.offset = 0

        print_blue("Initializated car controller")
        print_blue(f"Longitudinal params:")
        print_red(longitudinal_params)
        print_blue(f"Lateral params:")
        print_red(lateral_params)

    def set_offset(self, offset):
        """Changes the offset"""
        self.offset = offset
        
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
        target_speed, target_transform = self.reference[self.current_index][0], self.reference[self.current_index][1]
        current_location = current_transform.location
        current_yaw = np.deg2rad(current_transform.rotation.yaw)
        target_location = target_transform.location
        if self.offset != 0:
            # Displace the wp to the side
            r_vec = target_transform.get_right_vector()
            target_location_displaced = target_location + carla.Location(x=self.offset*r_vec.x,
                                                         y=self.offset*r_vec.y)
            #print("Displaced target location: ", target_location_displaced, "from: ", target_location, "offset: ", self.offset, "r_vec: ", r_vec)
        else:
            target_location_displaced = target_location
            
        #check if we need to advance the reference
        v1 = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
        if self.current_index < len(self.reference) - 1:
            next_target_location = self.reference[self.current_index + 1][1].location
            v2 = np.array([next_target_location.x - target_location.x, next_target_location.y - target_location.y])
            dot_product = np.dot(v1, v2)
            # print("Current location:", current_location)
            # print("Target location:", target_location)
            # print("Next target location:", next_target_location)
            # print("V1: ", v1)
            # print("V2: ", v2)
            # print("Dot product: ", dot_product)
            if dot_product < 0:
                #print("Advancing reference from: ", target_location, "to: ", next_target_location)
                self.current_index += 1
        else:
            self.done = True

        throttle_brake = self.longitudinal_controller.run_step(target_speed, current_speed)
        steering = self.lateral_controller.run_step(target_location=target_location_displaced, current_transform= current_transform)
        throttle = max(throttle_brake, 0)
        brake = max(-throttle_brake, 0)
        
        control = carla.VehicleControl(throttle=throttle, brake=brake, steer=steering)
        return control

