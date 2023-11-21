try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")

class KeyboardControl(object):
    def __init__(self, world):
        self.key_state = {
            "s": False,
            "l": False,
        }  # Track if 's' and 'l' keys are pressed
        self.stop_flag = False
        self.quit_flag = False

        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_flag = True
            elif event.type == pygame.KEYDOWN:
                if (event.key == K_ESCAPE) or (event.key == K_q and pygame.key.get_mods() & KMOD_CTRL):
                    self.quit_flag = True
                if event.key == pygame.K_s:
                    self.key_state["s"] = True
                elif event.key == pygame.K_l and self.key_state["s"]:
                    self.stop_flag = True  # Set stop_flag to True when 's' followed by 'l' is pressed
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_s:
                    self.key_state["s"] = False
                elif event.key == pygame.K_l:
                    self.key_state["l"] = False

        return self.quit_flag, self.stop_flag
