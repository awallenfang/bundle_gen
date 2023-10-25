import mitsuba as mi
import drjit as dr
import numpy as np

import matplotlib.pyplot as plt

import fiber
from brdf import TabulatedBCRDF

# Mitsuba prbvolpath

# interface Shape add method (shape.h)
# Look for similar method (eval_parameterization)
# edit cylinder
# add to bind_shape_generic in shape_v.cpp

# Make a method that computes the surface point in the direction

TEST = False

RAY_AMT = 10

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

    brdf = TabulatedBCRDF(["fiber_0/fiber_0_lambda" + str(i) + "_TM_depth6.binary" for i in range(24)])

    loop = mi.Loop("Tracing", lambda: (active, directions, origins, bounce_n, max_bounce))

    # while loop(active):
    #     # TODO: Somehow find which fiber it is
    #     ray = mi.Ray3f(origins, directions)
    #     intersection: mi.SurfaceInteraction3f = scene.ray_intersect(ray, active=active)
    #     bounding_box_hit = mi.Mask(intersection.t > 100000)

    #     output: mi.Shape = intersection.shape
    #     fiber_radius: mi.Float = output.eval_attribute_1("radius", intersection, active)
    #     fiber_dir: mi.Vector3f = output.eval_attribute("direction", intersection, active)
    #     dr.printf_async("%f", fiber_radius)
    #     new_ori, new_dir, new_mag = brdf.brdf(intersection, active, fiber_dir, fiber_radius,  sampler, 600.)

    #     origins[active] = new_ori
    #     directions[active] = new_dir
        
    #     bounce_n += 1
    #     active &= bounce_n < max_bounce
    #     active &= ~bounding_box_hit

    print(directions)
    print(origins)







else:
    rend_scene = fiber.preview_render_dict_from_fibers(fibers)
    print(rend_scene)

    img = mi.render(mi.load_dict(rend_scene))
    plt.imshow(img)
    plt.show()
