import mitsuba as mi

class Fiber():
    def __init__(self, pos_x: float, pos_y: float, radius: float, direction: list[float]):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.radius = radius
        self.direction = direction

def scene_dict_from_fibers(fibers: list[Fiber]) -> dict:
    scene_dict = {"type": "scene"}
    for (n, fiber) in enumerate(fibers):
        start_x = fiber.pos_x + 1000. * fiber.direction[0]
        start_y = fiber.pos_y + 1000. * fiber.direction[1]
        end_x = fiber.pos_x - 1000. * fiber.direction[0]
        end_y = fiber.pos_y - 1000. * fiber.direction[1]
        
        fiber_dict = {"type": "cylinder",
        "radius": fiber.radius,
        "p0": [start_x, start_y, -1000.],
        "p1": [end_x, end_y, 1000.]}

        scene_dict["fiber_" + str(n)] = fiber_dict
    return scene_dict




def preview_render_dict_from_fibers(fibers: list[Fiber]) -> dict:
    scene_dict = {"type": "scene",

    'integrator': {'type': 'path', 'max_depth': 8},
    'sensor': {'type': 'perspective',
        'fov': 35,
        'to_world': mi.ScalarTransform4f.look_at(
            origin=[50, -100, 30],
            target=[20, 0, 0],
            up=[0, 0, 1]
        ),
        'samples': {'type': 'independent', 'sample_count': 64}
    },
    'emitter': {'type': 'directional',
'direction': [1.0, 1.0, -1.0],
'irradiance': {
    'type': 'rgb',
    'value': 10.0,
}}

}
    for (n, fiber) in enumerate(fibers):
        start_x = fiber.pos_x + 1000. * fiber.direction[0]
        start_y = fiber.pos_y + 1000. * fiber.direction[1]
        end_x = fiber.pos_x - 1000. * fiber.direction[0]
        end_y = fiber.pos_y - 1000. * fiber.direction[1]
        
        fiber_dict = {"type": "cylinder",
        "radius": fiber.radius,
        "p0": [start_x, start_y, -1000.],
        "p1": [end_x, end_y, 1000.],
        'bsdf': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': [0.2, 0.25, 0.7]
            }
        }}

        scene_dict["fiber_" + str(n)] = fiber_dict
    return scene_dict