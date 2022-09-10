from turtle import distance
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

import math

res = 512
space_normalizer = np.array([res, res], dtype=float)
max_magnet_magnitude = 9000.0
magnet_force_constant = 200.0
max_magnet_speed = 1000.0

class PuckWorldMagnetEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None, human_player = False):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # simpler observation space: com and target com
        self.observation_space = spaces.Dict(
            {
                "com": spaces.Box(0.0, 1.0, shape=(2,), dtype=float),
                "vel": spaces.Box(float("-inf"), float("inf"), shape=(2,), dtype=float),
                "magnet_com": spaces.Box(0.0, 1.0, shape=(2,), dtype=float),
                "target": spaces.Box(0.0, 1.0, shape=(2,), dtype=float)
            }
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2, ), dtype=float) # standard deviation?
        self.human_player = human_player

        # Space
        self._space = pymunk.Space()
        self._space.gravity = (0.0, 0.0)

        # Physics
        # Time step
        self._dt = 1.0 / 120.0
        # Number of physics steps per screen frame
        self._physics_steps_per_frame = 4

        # pygame
        if self.render_mode == "human":
            pygame.init()
            self._screen = pygame.display.set_mode((res, res))
            self._clock = pygame.time.Clock()
            self._draw_options = pymunk.pygame_util.DrawOptions(self._screen)

    def _get_obs(self):
        com = np.array(self._ball.body.position) / space_normalizer
        vel = np.array(self._ball.body.velocity) / space_normalizer
        magnet_com = self.magnet_com / space_normalizer
        target = self.target_com / space_normalizer
        return {"com": com, "vel": vel, "magnet_com": magnet_com, "target": target}

    def _get_info(self, observation):
        displacement = np.array(observation["com"]) - np.array(observation["target"])
        distanceSquared = np.dot(displacement, displacement)
        distance = np.sqrt(distanceSquared) # note: this distance is already normalized
        return {"time": self._time, "distance": distance}

    def _get_reward(self, info):
        r_weight = 5.0
        reward = math.exp(-info["distance"] * r_weight) #tune with a constant weight
        return reward

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
        body.velocity = -50.0 + np.random.rand() * 100.0, -50.0 + np.random.rand() * 100.0
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.elasticity = 0.95
        shape.friction = 0.9
        self._space.add(body, shape)
        self._ball = shape

    def _create_target(self) -> None:
        self.target_radius = 50
        self.target_com = np.array([128.0 + np.random.rand() * 256.0, 128.0 + np.random.rand() * 256.0])

    def _create_magnet(self) -> None:
        self.magnet_radius = 50
        self.magnet_com = np.array([128.0 + np.random.rand() * 256.0, 128.0 + np.random.rand() * 256.0])
        self.magnet_vel = np.array([0.0, 0.0])

    def reset(self, seed = None, options = None):
        """
        Create the static bodies.
        :return: None
        """
        self._space = pymunk.Space()
        self._space.gravity = (0.0, 0.0)
        #self._space.damping = 0.5

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
        self._create_target()
        self._create_magnet()

        self._time = 0.0

        return self._get_obs()

    
    def step(self, action):
        
        if self.human_player == False:
            self.magnet_vel = action * max_magnet_speed

        # Progress time forward
        for x in range(self._physics_steps_per_frame):
            
            # timestep magnet
            magnet_com = np.array(self.magnet_com)
            magnet_com += self.magnet_vel * self._dt
            magnet_com = np.clip(magnet_com, np.array([0.0, 0.0]), space_normalizer)
            self.magnet_com = tuple(magnet_com)

            # apply magnet force to puck
            displacement = self.magnet_com - np.array(self._ball.body.position)
            distanceSquared = np.dot(displacement, displacement)
            distance = np.sqrt(distanceSquared)
            magnet_dir = displacement / distance if distance != 0.0 else np.array([0.0, 0.0])
            
            # our magnet force will scale linearly off distance to make control easier
            distanceNormed = distance / float(res)
            magnet_force_mag = magnet_force_constant * self._ball.body.mass * np.max([1.0 - distanceNormed, 0.0])
            magnet_force = tuple(magnet_dir * magnet_force_mag)
            self._ball.body.apply_force_at_local_point(magnet_force, (0.0, 0.0))
            

            self._space.step(self._dt)
            self._time += self._dt

        observation = self._get_obs()
        info = self._get_info(observation)
        reward = self._get_reward(info)
        done = self._time >= 5.0
        
        return observation, reward, done, info

    def render(self):
        if self.render_mode == "human":
            self._clear_screen()
            self._draw_objects()


            target_color = (0, 255, 0)
            pygame.draw.circle(self._screen, target_color, tuple(self.target_com), self.target_radius)

            pygame.draw.circle(self._screen, (0, 0, 0), tuple(self.magnet_com), self.magnet_radius)

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
                    self.magnet_vel = np.array([0.0, -100.0])
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                    self.magnet_vel = np.array([0.0, 100.0])
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                    self.magnet_vel = np.array([-100, 0.0])
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_d:
                    self.magnet_vel = np.array([100.0, 0.0])

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
        

        