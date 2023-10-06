import mitsuba as mi
import drjit as dr
import pprint
import numpy as np

import matplotlib.pyplot as plt

import fiber

# Mitsuba prbvolpath

TEST = False

RAY_AMT = 100

mi.set_variant('llvm_ad_rgb')

fibers = []
fibers.append(fiber.Fiber(0., 0., 3., [1.,0.,1.]))
fibers.append(fiber.Fiber(11., 0., 3., [0.,0.,1.]))
fibers.append(fiber.Fiber(22., 0., 3., [0.,0.,1.]))
fibers.append(fiber.Fiber(33., 0., 3., [0.,0.,1.]))
fibers.append(fiber.Fiber(44., 0., 3., [0.,0.,1.]))

def dummy(intersection, rand, active):
    pos = intersection.p * rand
    return (pos, intersection.to_world(intersection.wi))

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
    dirs = mi.Vector3f(1,0,0)


    sampler: mi.Sampler = mi.load_dict({'type': 'independent'})
    sampler.seed(213, RAY_AMT)    
    directions = mi.Vector3f(1,0,0)
    # dr.resize(directions, RAY_AMT)
    origins = mi.Point3f(-1,0,0)
    # dr.resize(origins, RAY_AMT)
    # origins = mi.Vector3f(-1,0,0)


    bounce_n = mi.UInt32(0)
    max_bounce = mi.UInt32(20)
    active: mi.Mask = mi.Mask(True)
    # dr.resize(active, RAY_AMT)

    loop = mi.Loop("Tracing", lambda: (active, directions, origins, bounce_n, max_bounce))

    while loop(active):
        ray = mi.Ray3f(origins, directions)
        # print(dr.width(ray.o.x), dr.width(ray.d.x))
        intersection = scene.ray_intersect(ray, active=active)

        rand = sampler.next_1d(active)
        new_ori, new_dir = dummy(intersection, rand, active)

        origins[active] = new_ori
        directions[active] = new_dir
        dr.printf_async("%f, %f, %f\n",directions.x, directions.y, directions.z)

        
        bounce_n += 1
        active &= bounce_n < max_bounce
        # active &= ~bounding_box_hit # < mi.Mask

    print(bounce_n)







else:
    rend_scene = fiber.preview_render_dict_from_fibers(fibers)
    print(rend_scene)
    # img = mi.render(mi.load_dict(test_scene))
    img = mi.render(mi.load_dict(rend_scene))
    plt.imshow(img)
    plt.show()
