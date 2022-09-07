import gym
from gym import spaces
import numpy as np
from typing import Optional

# Python imports
import random
from typing import List

# Library imports
import pygame

# pymunk imports
import pymunk
import pymunk.pygame_util


class RBGravityControl(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # simpler observation space: com and target com
        self.observation_space = spaces.Dict(
            {
                "com": spaces.Box(0.0, 1.0, shape=(2,), dtype=float),
                "target": spaces.Box(0.0, 1.0, shape=(2,), dtype=float)
            }
        )
        self.action_space = spaces.Box(low=-2000.0, high=2000.0, shape=(2, ), dtype=float)

        # Space
        self._space = pymunk.Space()
        self._space.gravity = (0.0, 900.0)

        # Physics
        # Time step
        self._dt = 1.0 / 60.0
        # Number of physics steps per screen frame
        self._physics_steps_per_frame = 1

        # pygame
        if self.render_mode == "human":
            pygame.init()
            self._screen = pygame.display.set_mode((512, 512))
            self._clock = pygame.time.Clock()
            self._draw_options = pymunk.pygame_util.DrawOptions(self._screen)

    def _get_obs(self):
        com = self._ball.body.position
        return {"com": np.array(com), "target": self.target_com}

    def _get_info(self):
        return {"score": self.score}

    def reset(self, seed = None, options = None):
        """
        Create the static bodies.
        :return: None
        """
        self.score = 0
        self._space.bodies.clear()

        static_body = self._space.static_body
        static_lines = [
            pymunk.Segment(static_body, (10, 10), (10, 502), 0.0),
            pymunk.Segment(static_body, (10, 502), (502, 502), 0.0),
            pymunk.Segment(static_body, (502, 502), (502, 10), 0.0),
            pymunk.Segment(static_body, (502, 10), (10, 10), 0.0),
        ]
        for line in static_lines:
            line.elasticity = 0.95
            line.friction = 0.9
        self._space.add(*static_lines)

        self._create_ball()
        self.create_target()

        

    
    def step(self, action):

        self._space.gravity = tuple(action)

        # Progress time forward
        for x in range(self._physics_steps_per_frame):
            self._space.step(self._dt)

        

        observation = self._get_obs()
        reward = 0
        done = False
        info = self._get_info()
        return observation, reward, done, info

    def render(self):
        if self.render_mode == "human":
            self._clear_screen()
            self._draw_objects()

            pygame.draw.circle(self._screen, (255, 0, 0), tuple(self.target_com), 50)
            pygame.display.flip()
            # Delay fixed time between frames
            self._clock.tick(50)
            pygame.display.set_caption("fps: " + str(self._clock.get_fps()))
            self._process_events()

    def _process_events(self) -> None:
        """
        Handle game and events like keyboard input. Call once per frame only.
        :return: None
        """
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_w:
                    self._space.gravity = (0.0, -900.0)
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                    self._space.gravity = (0.0, 900.0)
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                    self._space.gravity = (-900.0, 0.0)
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_d:
                    self._space.gravity = (900.0, 0.0)
        

    def _create_ball(self) -> None:
        """
        Create a ball.
        :return:
        """
        mass = 10
        radius = 75
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        body = pymunk.Body(mass, inertia)
        x = random.randint(115, 350)
        y = random.randint(115, 350)
        body.position = x, y
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.elasticity = 0.95
        shape.friction = 0.9
        self._space.add(body, shape)
        self._ball = shape

    def create_target(self) -> None:
        self.target_com = np.array([128.0 + np.random.rand() * 256.0, 128.0 + np.random.rand() * 256.0])
        

        

    def _clear_screen(self) -> None:
        """
        Clears the screen.
        :return: None
        """
        self._screen.fill(pygame.Color("white"))

    def _draw_objects(self) -> None:
        """
        Draw the objects.
        :return: None
        """
        self._space.debug_draw(self._draw_options)
        


if __name__ == "__main__":
    env = RBGravityControl("human")
    env.reset()

    while True:
        obs, rewards, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            obs = env.reset()