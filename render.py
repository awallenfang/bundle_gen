import mitsuba as mi
import drjit as dr
import pprint
import numpy as np

import matplotlib.pyplot as plt

import fiber
from brdf import brdf

# Mitsuba prbvolpath

TEST = False

RAY_AMT = 10000

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

# Check the interactions
if not TEST:
    scene: mi.Scene = mi.load_dict(fiber.scene_dict_from_fibers(fibers))

    dirs = mi.Vector3f(1,0,0)


    sampler: mi.Sampler = mi.load_dict({'type': 'independent'})
    sampler.seed(213, RAY_AMT)    

    directions = mi.Vector3f(1,0,0)
    origins = mi.Point3f(-1,0,0)
    magnitudes = mi.Float(1.)

    bounce_n = mi.UInt32(0)
    max_bounce = mi.UInt32(20)
    active: mi.Mask = mi.Mask(True)

    loop = mi.Loop("Tracing", lambda: (active, directions, origins, bounce_n, max_bounce))

    while loop(active):
        ray = mi.Ray3f(origins, directions)
        intersection: mi.SurfaceInteraction3f = scene.ray_intersect(ray, active=active)
        bounding_box_hit = mi.Mask(intersection.t > 100000)

        rand = sampler.next_1d(active)
        new_ori, new_dir, new_mag = brdf(intersection, rand, active)

        origins[active] = new_ori
        directions[active] = new_dir
        
        bounce_n += 1
        active &= bounce_n < max_bounce
        active &= ~bounding_box_hit

    print(directions)
    print(origins)







else:
    rend_scene = fiber.preview_render_dict_from_fibers(fibers)
    print(rend_scene)

    img = mi.render(mi.load_dict(rend_scene))
    plt.imshow(img)
    plt.show()
