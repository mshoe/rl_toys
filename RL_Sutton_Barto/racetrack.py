# Using https://gist.github.com/pat-coady/26fafa10b4d14234bfde0bb58277786d as a partial reference

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Optional

# Course descriptions
# W = wall (out of bounds)      = 0
# o = open space (in bounds)    = 1
# - = starting line cell        = 2
# + = finish line cell          = 3
tiny_course = ['WWWWWW',
               'Woooo+',
               'Woooo+',
               'WooWWW',
               'WooWWW',
               'WooWWW',
               'WooWWW',
               'W--WWW',]

big_course = ['WWWWWWWWWWWWWWWWWW',
              'WWWWooooooooooooo+',
              'WWWoooooooooooooo+',
              'WWWoooooooooooooo+',
              'WWooooooooooooooo+',
              'Woooooooooooooooo+',
              'Woooooooooooooooo+',
              'WooooooooooWWWWWWW',
              'WoooooooooWWWWWWWW',
              'WoooooooooWWWWWWWW',
              'WoooooooooWWWWWWWW',
              'WoooooooooWWWWWWWW',
              'WoooooooooWWWWWWWW',
              'WoooooooooWWWWWWWW',
              'WoooooooooWWWWWWWW',
              'WWooooooooWWWWWWWW',
              'WWooooooooWWWWWWWW',
              'WWooooooooWWWWWWWW',
              'WWooooooooWWWWWWWW',
              'WWooooooooWWWWWWWW',
              'WWooooooooWWWWWWWW',
              'WWooooooooWWWWWWWW',
              'WWooooooooWWWWWWWW',
              'WWWoooooooWWWWWWWW',
              'WWWoooooooWWWWWWWW',
              'WWWoooooooWWWWWWWW',
              'WWWoooooooWWWWWWWW',
              'WWWoooooooWWWWWWWW',
              'WWWoooooooWWWWWWWW',
              'WWWoooooooWWWWWWWW',
              'WWWWooooooWWWWWWWW',
              'WWWWooooooWWWWWWWW',
              'WWWW------WWWWWWWW']

class Racetrack(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "none"], "render_fps": 4}

    
    def __init__(self, course=tiny_course, max_speed=5, render_mode="human"):
        
        self.load_course(course)
        
        self.max_speed = max_speed
        self.cell_size = 16 # num pixels*pixels per cell
        self.window_width = self.cell_size * self.num_x
        self.window_height = self.cell_size * self.num_y
        self.render_mode = render_mode


        self.reset()

        # velocity may be changed by +1, -1, or 0 for both x and y components.
        # this leads to a total of 3x3 = 9 actions
        self.action_space = spaces.Discrete(9)

        self.observation_space = spaces.Dict(
            {
                "position": spaces.Box(low=np.array([0, 0]), high=np.array([self.num_x, self.num_y]), dtype=np.int),
                "velocity": spaces.Box(low=np.array([0, 0]), high=np.array([self.max_speed, self.max_speed]), dtype=np.int)
            }
        )

        self._action_to_acceleration = {
            0: np.array([-1, -1]),
            1: np.array([-1,  0]),
            2: np.array([-1,  1]),
            3: np.array([ 0, -1]),
            4: np.array([ 0,  0]),
            5: np.array([ 0,  1]),
            6: np.array([ 1, -1]),
            7: np.array([ 1,  0]),
            8: np.array([ 1,  1]),
        }

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode.
        """
        if self.render_mode == "human":
            import pygame  # import here to avoid pygame dependency with no render

            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
            self.clock = pygame.time.Clock()


    def _get_obs(self):
        return {"position": self._position, "velocity": self._velocity}


    def _get_info(self):
        return {"cell_type": self.get_cell_type(self._position)}


    def reset(self):
        # sample start from self.start_cells
        start_cell_index = np.random.randint(0, len(self.start_cells))
        self._position = self.start_cells[start_cell_index].copy()
        self._velocity = np.array([0, 0])

        return self._get_obs(), self._get_info()


    def step(self, action):
        
        acceleration = self._action_to_acceleration[action]

        """
        From the problem text:
        'Both velocity components are restricted to be nonnegative and less than 5,
        and they cannot both be zero except at the starting line'

        How we will handle these edge cases:
         - If a component ever goes below 0, set it to 0
         - If a component ever goes above 5, set it to 5
         - If both components go to zero, undo that move and pretend it was a [0, 0] acceleration,
            unless the car is on the starting line
        """
        next_velocity = self._velocity + acceleration
        
        next_velocity[0] = max(0, next_velocity[0])
        next_velocity[1] = max(0, next_velocity[1])

        next_velocity[0] = min(self.max_speed, next_velocity[0])
        next_velocity[1] = min(self.max_speed, next_velocity[1])

        if not np.any(next_velocity): # if both components are zero
            if self.get_cell_type(self._position) == 2:
                self._velocity = next_velocity
            else:
                self._velocity = self._velocity # doesn't change
        else:
            self._velocity = next_velocity

        """
        In the spirit of computer graphics, we do symplectic euler time integration :), 
        so we update velocity first, then position using the updated velocity.

        From the text:
        'If the care hits the track boundary, it is moved back to a random position on the starting line,
        both velocity components are reduced to zero, and the episode continues.'
        """
        self._position += self._velocity
        if self.get_cell_type(self._position) == 0:
            self._position = self.start_cells[np.random.randint(0, len(self.start_cells))].copy()
            self._velocity = np.array([0, 0])
        
        if self.get_cell_type(self._position) == 3:
            done = True
        else:
            done = False

        # the reward is -1 for every step
        reward = -1

        # the observation is the position and velocity
        observation = self._get_obs()
        info = self._get_info()

        if (self.render_mode == "human"):
            #print("action: ", self._action_to_acceleration[action])
            self.render()


        # return the 4-tuple (observation, reward, done, info)
        return observation, reward, done, info

    
    def render(self, mode='human'):
        
        mode = self.render_mode
        assert mode is not None  # The renderer will not call this function with no-rendering.
    
        import pygame # avoid global pygame dependency. This method is not called with no-render.
    
        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))
        

        for x in range(self.num_x):
            for y in range(self.num_y):
                cell_type = self.get_cell_type(x, y)
                color = (0, 0, 0)
                if cell_type == 0:
                    color = (0, 0, 0)
                elif cell_type == 1:
                    color = (255, 255, 255)
                elif cell_type == 2:
                    color = (255, 0, 0)
                elif cell_type == 3:
                    color = (0, 255, 0)
                
                pygame.draw.rect(
                    canvas,
                    color,
                    pygame.Rect(
                        (x * self.cell_size, (self.num_y-1-y) * self.cell_size),
                        (self.cell_size, self.cell_size)
                    )
                )

        # draw the car:
        pygame.draw.rect(
            canvas,
            (0, 0, 255),
            pygame.Rect(
                (self._position[0] * self.cell_size, (self.num_y-1-self._position[1]) * self.cell_size),
                (self.cell_size, self.cell_size)
            )
        )

        # Finally, add some gridlines
        for y in range(1, self.num_y):
            pygame.draw.line(
                canvas,
                0,
                (0, self.cell_size * y),
                (self.window_width, self.cell_size * y),
                width=2,
            )
        for x in range(1, self.num_x):
            pygame.draw.line(
                canvas,
                0,
                (self.cell_size * x, 0),
                (self.cell_size * x, self.window_height),
                width=2
            )

        if mode == "human":
            assert self.window is not None
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])


    def load_course(self, course):
        self.num_y = len(course)
        self.num_x = len(course[0])

        # Load data from python list to numpy array
        self.grid = np.zeros(shape=(self.num_x, self.num_y), dtype=int)
        for x in range(self.num_x):
            for y in range(self.num_y):
                cell_char = course[y][x]
                if cell_char == 'W':
                    self.grid[x, y] = 0
                elif cell_char == 'o':
                    self.grid[x, y] = 1
                elif cell_char == '-':
                    self.grid[x, y] = 2
                elif cell_char == '+':
                    self.grid[x, y] = 3

        # keep track of the start locations
        self.start_cells = []
        for x in range(self.num_x):
            for y in range(self.num_y):
                if self.get_cell_type(x, y) == 2:
                    self.start_cells.append(np.array([x, y]))


    def get_cell_type(self, x, y=None):
        # this function exists to flip the grid vertically, so y = 0 corresponds to the bottom of the grid
        # It also ensures that we don't index outside of the grid bounds

        if y is None: # if y is not provided, then x is a tuple (function overloading)
            yy = x[1]
            xx = x[0]
        else:
            xx = x
            yy = y
        if xx < 0 or xx >= self.num_x or yy < 0 or yy >= self.num_y:
            return 0
        return self.grid[xx, self.num_y-1 - yy]

    
