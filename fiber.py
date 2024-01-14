import math, statistics

import mitsuba as mi
import scipy.stats.qmc as qmc
import numpy as np

class Fiber():
    def __init__(self, pos_x: float, pos_y: float, radius: float, direction: list[float]):
        assert(len(direction) == 3)
        
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.radius = radius
        self.direction = direction
        self.mitsuba_direction = mi.Vector3f(direction)

def scene_dict_from_fibers(fibers: list[Fiber]) -> dict:
    scene_dict = {"type": "scene"}

    for (n, fiber) in enumerate(fibers):
        start_x = fiber.pos_x + 100000000. * fiber.direction[0]
        start_y = fiber.pos_y + 100000000. * fiber.direction[1]
        end_x = fiber.pos_x - 100000000. * fiber.direction[0]
        end_y = fiber.pos_y - 100000000. * fiber.direction[1]
        
        fiber_dict = {"type": "cylinder",
        "radius": fiber.radius,
        "p0": [start_x, start_y, 100000000.],
        "p1": [end_x, end_y, -100000000.],}


        scene_dict["fiber_" + str(n)] = fiber_dict
        
    return scene_dict

def get_bounds(fibers: list[Fiber]):
    center_x = (max(map(lambda f: f.pos_x, fibers)) - min(map(lambda f: f.pos_x, fibers))) / 2.
    center_y = (max(map(lambda f: f.pos_y, fibers)) - min(map(lambda f: f.pos_y, fibers))) / 2.
    max_radius = max(map(lambda f: math.sqrt((f.pos_x - center_x)**2 + (f.pos_y - center_y) ** 2), fibers))

    return max_radius + fibers[0].radius, center_x, center_y

def min_distance_between_fibers(fibers: list[Fiber]) -> float:
    min_dist = 99999999999999999999999999999999999999999.

    for fiber in fibers:
        for other in fibers:
            dist = math.sqrt((fiber.pos_x - other.pos_x) ** 2 + (fiber.pos_y - other.pos_y) ** 2)

            if dist < min_dist:
                min_dist = dist
    return min_dist




def preview_render_dict_from_fibers(fibers: list[Fiber]) -> dict:
    scene_dict = {"type": "scene",

    'integrator': {'type': 'path', 'max_depth': 8},
    'sensor': {'type': 'perspective',
        'fov': 35,
        'to_world': mi.ScalarTransform4f.look_at(
            origin=[-20, -100, 10],
            target=[0, 0, 0],
            up=[0, 0, 1]
        ),
        'samples': {'type': 'independent', 'sample_count': 64}
    },
    'emitter': {'type': 'constant',
'radiance': {
    'type': 'rgb',
    'value': 0.7,
}}

}
    for (n, fiber) in enumerate(fibers):
        start_x = fiber.pos_x + 100000000. * fiber.direction[0]
        start_y = fiber.pos_y + 100000000. * fiber.direction[1]
        end_x = fiber.pos_x - 100000000. * fiber.direction[0]
        end_y = fiber.pos_y - 100000000. * fiber.direction[1]
        
        fiber_dict = {"type": "cylinder",
        "radius": fiber.radius,
        "p0": [start_x, start_y, -100000000.],
        "p1": [end_x, end_y, 100000000.],
        'bsdf': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': [0.2, 0.25, 0.7]
            }
        }}

        scene_dict["fiber_" + str(n)] = fiber_dict
    return scene_dict

def generate_random(fiber_radius, bundle_radius) -> list[Fiber]:
    poisson_engine = qmc.PoissonDisk(d=2, radius=(2.*fiber_radius)/bundle_radius)
    samples = poisson_engine.random(1000000)
    samples -= 0.5
    samples *= 2.*bundle_radius
    samples = samples[np.sqrt(samples[:,0]*samples[:,0] + samples[:,1] * samples[:,1]) <= bundle_radius]

    fibers = []
    for elem in samples:
        fibers.append(Fiber(elem[0], elem[1], fiber_radius, [0.,0.,1.]))
    return fibers

def generate_single(radius) -> list[Fiber]:
    fibers = []
    fibers.append(Fiber(0.,0.,radius, [0.,0.,1.]))
    return fibers