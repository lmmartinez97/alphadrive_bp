from dataclasses import dataclass
import sys
import glob

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

@dataclass
class VehicleState:
    """
    Represents the state of a vehicle in the simulation at a specific frame.

    Attributes:
        actor_id (int): Unique identifier for the vehicle actor.
        position (carla.Location): Current position of the vehicle.
        velocity (carla.Vector3D): Current velocity of the vehicle.
        acceleration (carla.Vector3D): Current acceleration of the vehicle.
        rotation (carla.Rotation): Current rotation of the vehicle.
        angular_velocity (carla.Vector3D): Current angular velocity of the vehicle.
        frame_counter (int): Frame number when the state was recorded.
        throttle (float): Current throttle input applied to the vehicle.
        brake (float): Current brake input applied to the vehicle.
        steer (float): Current steering input applied to the vehicle.
        handbrake (bool): Indicates if the handbrake is engaged.
        reverse (bool): Indicates if the vehicle is in reverse gear.
        manual_gear_shift (bool): Indicates if manual gear shifting is enabled.
        gear (int): Current gear of the vehicle.
    """
    actor_id: int
    position: carla.Location
    velocity: carla.Vector3D
    acceleration: carla.Vector3D
    rotation: carla.Rotation
    angular_velocity: carla.Vector3D
    frame_counter: int
    throttle: float
    brake: float
    steer: float
    handbrake: bool
    reverse: bool
    manual_gear_shift: bool
    gear: int
    # Other necessary vehicle attributes

class StateManager:
    """
    Manages the state of vehicles in the simulation.

    Attributes:
        vehicle_states (list): List to store VehicleState objects.
    """
    def __init__(self):
        self.vehicle_states = []

    def save_vehicle_state(self, vehicle, frame_number):
        """
        Records the state of a vehicle at the specified frame number.

        Args:
            vehicle (carla.Vehicle): The vehicle whose state is to be saved.
            frame_number (int): The frame number when the state is recorded.
        """
        state = VehicleState(
            actor_id=vehicle.id,
            position=vehicle.get_location(),
            velocity=vehicle.get_velocity(),
            acceleration=vehicle.get_acceleration(),
            rotation=vehicle.get_transform().rotation,
            angular_velocity=vehicle.get_angular_velocity(),
            frame_counter=frame_number,
            throttle=vehicle.get_control().throttle,
            brake=vehicle.get_control().brake,
            steer=vehicle.get_control().steer,
            handbrake=vehicle.get_control().handbrake,
            reverse=vehicle.get_control().reverse,
            manual_gear_shift=vehicle.get_control().manual_gear_shift,
            gear=vehicle.get_control().gear
        )
        self.vehicle_states.append(state)

    def restore_vehicle_state(self, vehicle, target_frame_number):
        """
        Restores the state of a vehicle to the specified frame.

        Args:
            vehicle (carla.Vehicle): The vehicle to restore the state.
            target_frame_number (int): The frame number to restore the state.
        """
        matching_states = [
            state for state in self.vehicle_states
            if state.actor_id == vehicle.id and state.frame_counter == target_frame_number
        ]

        if matching_states:
            state_to_restore = matching_states[0]  # Assuming there's only one match
            vehicle.set_location(state_to_restore.position)
            vehicle.set_velocity(state_to_restore.velocity)
            vehicle.set_acceleration(state_to_restore.acceleration)
            vehicle.set_transform(
                carla.Transform(state_to_restore.position, state_to_restore.rotation)
            )
            vehicle.set_angular_velocity(state_to_restore.angular_velocity)
            vehicle.apply_control(carla.VehicleControl(
                throttle=state_to_restore.throttle,
                brake=state_to_restore.brake,
                steer=state_to_restore.steer,
                handbrake=state_to_restore.handbrake,
                reverse=state_to_restore.reverse,
                manual_gear_shift=state_to_restore.manual_gear_shift,
                gear=state_to_restore.gear
                # Set other necessary control attributes here
            ))
            # Restore other necessary vehicle attributes here
        else:
            print(f"No matching VehicleState found for actor {vehicle.id} at frame {target_frame_number}")
            # Handle if no matching state is found (e.g., raise an exception or log a message)
