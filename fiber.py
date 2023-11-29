import math, statistics

import mitsuba as mi

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
    center_x = statistics.mean(map(lambda f: f.pos_x, fibers))
    center_y = statistics.mean(map(lambda f: f.pos_y, fibers))
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