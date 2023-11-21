import glob
import numpy as np
import os
import sys
import weakref
try:
    import pygame
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")

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
    print("CARLA egg not found")
    sys.exit(1)
import carla
from carla import ColorConverter as cc

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    """Class for camera management"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (
                carla.Transform(
                    carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)
                ),
                attachment.SpringArm,
            ),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), attachment.Rigid),
            (
                carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)),
                attachment.SpringArm,
            ),
            (
                carla.Transform(
                    carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)
                ),
                attachment.SpringArm,
            ),
            (
                carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)),
                attachment.Rigid,
            ),
        ]
        self.transform_index = 1
        self.sensors = [
            ["sensor.camera.rgb", cc.Raw, "Camera RGB"],
            ["sensor.camera.depth", cc.Raw, "Camera Depth (Raw)"],
            ["sensor.camera.depth", cc.Depth, "Camera Depth (Gray Scale)"],
            [
                "sensor.camera.depth",
                cc.LogarithmicDepth,
                "Camera Depth (Logarithmic Gray Scale)",
            ],
            [
                "sensor.camera.semantic_segmentation",
                cc.Raw,
                "Camera Semantic Segmentation (Raw)",
            ],
            [
                "sensor.camera.semantic_segmentation",
                cc.CityScapesPalette,
                "Camera Semantic Segmentation (CityScapes Palette)",
            ],
            ["sensor.lidar.ray_cast", None, "Lidar (Ray-Cast)"],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            if item[0].startswith("sensor.camera"):
                blp.set_attribute("image_size_x", str(hud.dim[0]))
                blp.set_attribute("image_size_y", str(hud.dim[1]))
            elif item[0].startswith("sensor.lidar"):
                blp.set_attribute("range", "50")
            item.append(blp)
        self.index = None

    def toggle_camera(self):
        """Activate a camera"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """Set a sensor"""
        index = index % len(self.sensors)
        needs_respawn = (
            True
            if self.index is None
            else (
                force_respawn or (self.sensors[index][0] != self.sensors[self.index][0])
            )
        )
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1],
            )

            # We need to pass the lambda a weak reference to
            # self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(
                lambda image: CameraManager._parse_image(weak_self, image)
            )
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        self.hud.notification("Recording %s" % ("On" if self.recording else "Off"))

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith("sensor.lidar"):
            points = np.frombuffer(image.raw_data, dtype=np.dtype("f4"))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(
                lidar_data
            )  # pylint: disable=assignment-from-no-return
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk("_out/%08d" % image.frame)


# ==============================================================================
# -- StaticCamera -------------------------------------------------------------
# ==============================================================================


class StaticCamera(object):
    def __init__(self, transform, world, width, height):
        self.sensor = None
        self.surface = None
        self.recording = False
        self.transform = transform
        self.world = world
        self.width = width
        self.height = height
        self.sensors = [
            ["sensor.camera.rgb", cc.Raw, "Camera RGB"],
            ["sensor.camera.depth", cc.Raw, "Camera Depth (Raw)"],
            ["sensor.camera.depth", cc.Depth, "Camera Depth (Gray Scale)"],
            [
                "sensor.camera.depth",
                cc.LogarithmicDepth,
                "Camera Depth (Logarithmic Gray Scale)",
            ],
            [
                "sensor.camera.semantic_segmentation",
                cc.Raw,
                "Camera Semantic Segmentation (Raw)",
            ],
            [
                "sensor.camera.semantic_segmentation",
                cc.CityScapesPalette,
                "Camera Semantic Segmentation (CityScapes Palette)",
            ],
            ["sensor.lidar.ray_cast", None, "Lidar (Ray-Cast)"],
        ]
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            if item[0].startswith("sensor.camera"):
                blp.set_attribute("image_size_x", str(self.width))
                blp.set_attribute("image_size_y", str(self.height))
            elif item[0].startswith("sensor.lidar"):
                blp.set_attribute("range", "50")
            item.append(blp)
        self.index = 0

    def set_sensor(self, notify=True, force_respawn=False):
        """Set a sensor"""
        self.sensor = self.world.spawn_actor(
            self.sensors[self.index][-1], self.transform
        )

        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: StaticCamera._parse_image(weak_self, image))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(self.sensors[self.index][1])
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk("_out/%08d" % image.frame)

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (self.width, 0))