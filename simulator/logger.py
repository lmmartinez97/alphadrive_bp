

# ==============================================================================
# -- Data logger -------------------------------------------------------
# ==============================================================================


class DataLogger:
    def __init__(self, ego_actor, world):
        actor_list = world.world.get_actors()
        self.vehicle_list = []
        self.vehicle_list.append(ego_actor)
        self.lane_invasion_sensor = world.lane_invasion_sensor
        self.collision_sensor = world.collision_sensor

        self.vehicle_list = self.vehicle_list + world.players

        self.episode_data = []
        self.vehicle_name_list = ["ego_vehicle"]
        for i in range(len(self.vehicle_list)):
            self.vehicle_name_list.append("vehicle_" + str(i + 1).zfill(2))

    def update_actors(self, ego_actor, world):
        self.vehicle_list.clear()
        self.vehicle_list.append(ego_actor)
        self.lane_invasion_sensor = world.lane_invasion_sensor
        self.collision_sensor = world.collision_sensor

        self.vehicle_list = self.vehicle_list + world.players

        return 0

    # def log_vehicle_data(self, vehicle_list):
    def log_vehicle_data(self):
        self.vehicle_data = {}
        for idx, vehicle in enumerate(self.vehicle_list):
            t = vehicle.get_transform()
            v = vehicle.get_velocity()
            w = vehicle.get_angular_velocity()
            a = vehicle.get_acceleration()
            c = vehicle.get_control()

            log_dict = {
                "x": t.location.x,
                "y": t.location.y,
                "z": t.location.z,
                "pitch": t.rotation.pitch,
                "roll": t.rotation.roll,
                "yaw": t.rotation.yaw,
                "vx": v.x,
                "vy": v.y,
                "vz": v.z,
                "wx": w.x,
                "wy": w.y,
                "wz": w.z,
                "ax": a.x,
                "ay": a.y,
                "az": a.z,
                "current_steer": c.steer,
                "current_throttle": c.throttle,
                "current_brake": c.brake,
            }
            for key, value in log_dict.items():
                log_dict[key] = round(value, 5)

            update_dict = {self.vehicle_name_list[idx]: log_dict}

            self.vehicle_data.update(update_dict)

        return self.vehicle_data

    def log_sensor_data(self):
        # The simulation can retrieve a list with the lanes the ego_vehicle has crossed, which can be of length greater than 1.
        # Since every lane crossing will be a termination condition for the training we only care about the length of the list, not it's contents, which allows to send a fixed size object
        ret_sensor_data = {
            "lanes_crossed": len(self.lane_invasion_sensor.lane_crossed),
            "collision_against": len(self.collision_sensor.collision_object),
        }

        return ret_sensor_data

    #   def log_data(self, vehicle_list):
    def log_data(self):
        # vehicle_data = self.log_vehicle_data(vehicle_list)
        self.vehicle_data = self.log_vehicle_data()
        self.sensor_data = self.log_sensor_data()

        return self.vehicle_data, self.sensor_data

    def log_to_file(self, frame_number, episode_number, action=None):
        dict_ = {
            "ep_frame": frame_number,
            "vehicle_data": self.vehicle_data,
            "sensor_data": self.sensor_data,
            "action": action,
        }
        file = "./client_log/episode_" + str(episode_number).zfill(5) + ".json"
        print("Logging to file: {}".format(file))
        with open(file, "a") as f:
            json.dump(dict_, f)
            f.write("\n")