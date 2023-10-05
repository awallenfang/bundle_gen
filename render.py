import mitsuba as mi
import drjit as dr
from drjit.llvm import Loop, UInt32, Array3f
import pprint
import numpy as np

import matplotlib.pyplot as plt

import fiber

TEST = False

RAY_AMT = 100

mi.set_variant('scalar_rgb')

fibers = []
fibers.append(fiber.Fiber(0., 0., 3., [1.,0.,1.]))
fibers.append(fiber.Fiber(11., 0., 3., [0.,0.,1.]))
fibers.append(fiber.Fiber(22., 0., 3., [0.,0.,1.]))
fibers.append(fiber.Fiber(33., 0., 3., [0.,0.,1.]))
fibers.append(fiber.Fiber(44., 0., 3., [0.,0.,1.]))

# Create BRDF from data

# Send em into the scene
# Check the interactions
# Do x bounces
# Once it goes out of bounds record its direction
if not TEST:
    scene = mi.load_dict(fiber.scene_dict_from_fibers(fibers))

    # origin = mi.Point3f()
    # origin.x = -1
    # origin.y = 0
    # origin.z = 0

    # direction = mi.Vector3f()
    # direction.x = 1
    # direction.y = 0
    # direction.z = 0
    # rays = [mi.Ray3f(origin, direction) for _ in range(RAY_AMT)]

    directions = Array3f([1 for _ in range(RAY_AMT)], [0 for _ in range(RAY_AMT)], [0 for _ in range(RAY_AMT)])
    origins = Array3f([-1 for _ in range(RAY_AMT)], [0 for _ in range(RAY_AMT)], [0 for _ in range(RAY_AMT)])

    print(dr.gather(Array3f, directions, 0))

    bounce_n = UInt32(0)
    max_bounce = UInt32(20)

    loop = Loop("Tracing", lambda: (directions, origins, bounce_n, max_bounce))

    while loop(bounce_n < max_bounce):
        
        for ray_n in range(RAY_AMT):
            ray = mi.Ray3f()
            intersection = scene.ray_intersect(ray)
        bounce_n += 1

    print(bounce_n)
else:
    rend_scene = fiber.preview_render_dict_from_fibers(fibers)
    print(rend_scene)
    # img = mi.render(mi.load_dict(test_scene))
    img = mi.render(mi.load_dict(rend_scene))
    plt.imshow(img)
    plt.show()
