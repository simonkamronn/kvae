import pygame
import pymunk.pygame_util
import numpy as np
import os


class Cannon:
    def __init__(self, dt=0.2, res=(32, 32), init_pos=(3, 3), wall=None):
        pygame.init()

        self.dt = dt
        self.res = res
        self.screen = pygame.display.set_mode(res, 0, 8)
        self.gravity = (0.0, -9.81)
        self.initial_position = init_pos
        self.space = pymunk.Space()
        self.space.gravity = self.gravity
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.clock = pygame.time.Clock()
        self.wall = wall
        self.static_lines = []

    def _clear(self):
        self.screen.fill(pygame.color.THECOLORS["black"])

    def add_wall(self, p):
        self.static_lines = []
        if (self.wall is not None) and (np.random.random_sample() < p):
            if isinstance(self.wall, tuple):
                self.static_lines.append(pymunk.Segment(self.space.static_body, self.wall[0], self.wall[1], 1.0))
            elif callable(self.wall):
                x, y = self.wall()
                self.static_lines.append(pymunk.Segment(self.space.static_body, (x[0], y[0]), (x[1], y[1]), 1.0))

        # Add floor
        self.static_lines.append(pymunk.Segment(self.space.static_body, (0, 1), (self.res[1], 1), 0.0))

        # Add roof
        self.static_lines.append(pymunk.Segment(self.space.static_body, (0, self.res[1]), (self.res[1], self.res[1]), 0.0))

        # Set properties
        for line in self.static_lines:
            line.elasticity = 1.0
            line.friction = 0.9
            line.color = pygame.color.THECOLORS["white"]
        self.space.add(self.static_lines)
        return True

    def create_ball(self, radius=3):
        inertia = pymunk.moment_for_circle(1, 0, radius, (0, 0))
        body = pymunk.Body(1, inertia)
        body.position = self.initial_position() if callable(self.initial_position) else self.initial_position

        shape = pymunk.Circle(body, radius, (0, 0))
        shape.elasticity = 1.0
        shape.friction = 0.0
        shape.color = pygame.color.THECOLORS["white"]
        return shape

    def fire(self, angle=50, velocity=20, radius=3):
        speedX = velocity * np.cos(angle * np.pi / 180)
        speedY = velocity * np.sin(angle * np.pi / 180)

        ball = self.create_ball(radius)
        ball.body.velocity = (speedX, speedY)

        self.space.add(ball, ball.body)
        return ball

    def fire_constrained(self, range_limits, height_limits, radius):
        range_max = np.random.uniform(*range_limits)
        height_max = np.random.uniform(*height_limits)
        angle, velocity = self.get_init_values(range_max, height_max, self.gravity)

        return self.fire(angle, velocity, radius)

    def get_init_values(self, range_max, height_max, gravity):
        # https://en.wikipedia.org/wiki/Projectile_motion#Relation_between_horizontal_range_and_maximum_height
        # h=R*tan(theta)/4
        theta = np.arctan(4.0 * height_max / range_max)
        # https://en.wikipedia.org/wiki/Range_of_a_projectile
        # R=v^2/g * sin(2 * theta)
        muzzle_velocity = np.sqrt(range_max*np.abs(gravity[1])/np.sin(2 * theta))
        return np.degrees(theta), muzzle_velocity

    def run(self, iterations=20, sequences=500, range_limits=(15, 28), height_limits=(10, 25), radius=3,
            flip_gravity=None, save=None, filepath='../../data/balls.npz', delay=None):
        if save:
            data = np.empty((sequences, iterations, *self.res), dtype=np.float32)
        controls = np.empty((2, sequences))
        for s in range(sequences):
            # Maybe add wall
            wall = self.add_wall(0.8)

            ball = self.fire_constrained(range_limits, height_limits, radius)
            for i in range(iterations):
                self._clear()
                self.space.debug_draw(self.draw_options)
                self.space.step(self.dt)
                pygame.display.flip()

                if delay:
                    self.clock.tick(delay)

                if save == 'png':
                    pygame.image.save(self.screen, os.path.join(filepath, "bouncing_balls_%02d_%02d.png" % (s, i)))
                elif save == 'npz':
                    data[s, i] = pygame.surfarray.array2d(self.screen).swapaxes(1, 0) / 255

                if flip_gravity and ball.body.position.x > flip_gravity:
                    self.space.gravity = (self.gravity[0], -self.gravity[1])

            # Remove the ball and the wall from the space
            self.space.remove(ball, ball.body)
            if wall:
                self.space.remove(self.static_lines)
            if flip_gravity:
                self.space.gravity = self.gravity

        if save == 'npz':
            np.savez(os.path.abspath(filepath), images=data, controls=controls)


if __name__ == '__main__':
    """Experiment configurations

    Random x-position: wall=lambda: (np.tile(np.random.randint(30*scale, 47*scale), 2), (0, 48*scale))
    """
    scale = 1
    experiment = 'random_wall'

    # Create data dir
    if not os.path.exists('../../data'):
        os.makedirs('../../data')

    if experiment == 'normal':
        cannon = Cannon(dt=0.2, res=(32*scale, 32*scale), init_pos=(4*scale, 4*scale), wall=None)
        cannon.run(delay=None, iterations=20, sequences=500, radius=4*scale, range_limits=(10*scale, 25*scale),
                   height_limits=(10*scale, 25*scale), filepath='../../data/images/', save='png')
    elif experiment == 'wall':
        cannon = Cannon(dt=0.2, res=(40*scale, 40*scale), init_pos=(4*scale, 4*scale), wall=((20*scale, 32*scale),
                                                                                             (20*scale, 10*scale)))
        cannon.run(delay=None, iterations=20, sequences=500, radius=4*scale, range_limits=(10*scale, 25*scale),
                   height_limits=(10*scale, 22*scale), filepath='../../data/balls_wall.npz', save='npz')
    elif experiment == 'gravity':
        cannon = Cannon(dt=0.2, res=(32*scale, 32*scale), init_pos=(4*scale, 4*scale), wall=None)
        cannon.run(delay=None, iterations=20, sequences=500, radius=4*scale, range_limits=(15*scale, 25*scale),
                   flip_gravity=15*scale,
                   height_limits=(10*scale, 20*scale), filepath='../../data/balls_gravity.npz', save='npz')

    elif experiment == 'random_wall':
        cannon = Cannon(dt=0.2,
                        res=(48*scale, 48*scale),
                        init_pos=lambda: (np.random.randint(1, 20)*scale, np.random.randint(1, 29)*scale),
                        wall=lambda: (np.tile(np.random.randint(30 * scale, 47 * scale), 2), (0, 48 * scale)))

        cannon.run(delay=0,
                   iterations=20*scale,
                   sequences=5000,
                   radius=4*scale,
                   range_limits=(10*scale, 48*scale),
                   height_limits=(10*scale, 30*scale),
                   filepath='../../data/balls_random_wall.npz',
                   save='npz')

