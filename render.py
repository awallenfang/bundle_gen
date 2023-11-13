import mitsuba as mi
import drjit as dr
import numpy as np
import time

import matplotlib.pyplot as plt

import fiber
from brdf import TabulatedBCRDF, cartesian_to_polar

def cartesian_to_polar(x,y,z):
    phi = np.arctan2(y,x)
    theta = np.arccos(z / np.sqrt(x*x + y*y + z*z))

    return (theta, phi)

def plot_results(directions, magnitudes):
    out_model = np.zeros((OUT_THETA, OUT_PHI))

    phis = np.arctan2(directions[:,1], directions[:,0])
    thetas = np.arccos(directions[:,2] / np.linalg.norm(directions, axis=1))

    # phis[phis<0] += np.pi*2
    # phi_coords = np.floor((phis / (2*np.pi)) * OUT_PHI)
    # phi_coords = phi_coords.astype(int)
    # print(np.max(phi_coords))

    # print(f'phi:0 -> {np.floor((0 / (2*np.pi)) * OUT_PHI)}')
    # print(f'theta:0 -> {np.floor((0 / np.pi) * OUT_THETA)}')

    # theta_coords = np.floor((thetas / np.pi) * OUT_THETA)
    # theta_coords = theta_coords.astype(int)
    # # print(np.max(theta_coords))

    # print(thetas)
    for i in range(0, RAY_AMT):
        dir = directions[i]
        phi, theta = (phis[i], thetas[i])
        pos_x = int(np.floor(((phi + 2*np.pi * (phi < 0)) / (np.pi*2)) * OUT_PHI))
        pos_y = int(np.floor((theta / (np.pi)) * OUT_THETA))
   
        mag = magnitudes[i]

        out_model[pos_y, pos_x] += mag / np.sin(theta)

    fig, ax = plt.subplots(1)
    ax.set_xlabel("phi")
    ax.set_ylabel("theta")
    ax.imshow(out_model)
    plt.show()

def render_structure(fibers, brdf):
    scene: mi.Scene = mi.load_dict(fiber.scene_dict_from_fibers(fibers))
    sampler: mi.Sampler = mi.load_dict({'type': 'independent'})
    seed = int(time.time())
    sampler.seed(seed, RAY_AMT)    
    # sampler.seed(3, RAY_AMT)    

    directions = mi.Vector3f(1,0,0)
    origins = mi.Point3f(-20,0,0)
    magnitudes = mi.Float(1.)

    bounce_n = mi.UInt32(0)
    max_bounce = mi.UInt32(5)
    active: mi.Mask = mi.Mask(True)


    loop = mi.Loop("Tracing", lambda: (active, directions, origins, magnitudes,  bounce_n, max_bounce))

    while loop(active):
        ray = mi.Ray3f(origins, directions)
        intersection: mi.SurfaceInteraction3f = scene.ray_intersect(ray, active=active)

        # Check if the ray is valid before any brdfs are run
        t_too_big = mi.Mask(intersection.t > 999999999)
        t_too_small = mi.Mask(intersection.t < 0)
        active &= ~t_too_big
        active &= ~t_too_small

        dr.printf_async("Ori: (%f,%f,%f)\n", origins.x, origins.y, origins.z)
        dr.printf_async("Dir: (%f,%f,%f)\n", directions.x, directions.y, directions.z)
        dr.printf_async("t: %f\n", intersection.t)

        output: mi.Shape = intersection.shape
        new_ori, new_dir, new_mag = brdf.brdf(intersection, active, sampler, 414.)

        origins[active] = new_ori
        directions[active] = new_dir
        magnitudes[active] *= new_mag
        bounce_n[active] += 1
        active &= bounce_n < max_bounce
        active &= magnitudes <= 0.000001
    print(dr.max(bounce_n))

    n_dir = directions.numpy()
    n_mag = magnitudes.numpy()

    return (n_dir, n_mag)

# length = r/ (sqrt(1 - dot(out, fiber_dir)^2))
# wenn dot nahe 1 wegwerfen

# Check code where exactly it's written if it's even uniform
# Get the test cases to work correctly

TEST = False

RAY_AMT = 1
OUT_PHI = 100
OUT_THETA = 100

THETA_TABLE_SIZE=450
PHI_TABLE_SIZE=880

mi.set_variant('llvm_ad_rgb')

fibers = []
fibers.append(fiber.Fiber(0., 0., 3., [0.,0.,1.]))
# fibers.append(fiber.Fiber(10., 0., 3., [0.,0.,1.]))
# fibers.append(fiber.Fiber(10., 10., 3., [0.,0.,1.]))
# fibers.append(fiber.Fiber(0., 10., 3., [0.,0.,1.]))
# fibers.append(fiber.Fiber(10., 0., 3., [0.,0.,1.]))
# fibers.append(fiber.Fiber(-10., -10., 3., [0.,0.,1.]))
# fibers.append(fiber.Fiber(0., -10., 3., [0.,0.,1.]))
# fibers.append(fiber.Fiber(-10., 0., 3., [0.,0.,1.]))

# Check the interactions
if not TEST:
    brdf = TabulatedBCRDF(["fiber_0/fiber_0_lambda" + str(i) + "_TM_depth6.binary" for i in range(24)])

    # phi = np.linspace(0.,1.,THETA_TABLE_SIZE).reshape(THETA_TABLE_SIZE,1)
    # intensities = np.ones((THETA_TABLE_SIZE, PHI_TABLE_SIZE)) * phi
    # intensities = np.zeros((THETA_TABLE_SIZE, PHI_TABLE_SIZE))
    # intensities[0:50,:] = 1
    # # intensities[100:150,:] = 1
    # # intensities[200:250,:] = 1
    # # intensities[300:350,:] = 1
    # intensities[400:450,:] = 1

    # intensities[:,0:50] = 1
    # # intensities[:,100:150] = 1
    # # intensities[:,200:250] = 1
    # # intensities[:,300:350] = 1
    # # intensities[:,400:450] = 1
    # # intensities[:,500:550] = 1
    # # intensities[:,600:650] = 1
    # # intensities[:,700:750] = 1
    # intensities[:,800:850] = 1

    # brdf.set_test_intensities(intensities)
    # brdf.show_layers()
    directions, magnitudes = render_structure(fibers, brdf)

    plot_results(directions, magnitudes)

else:
    rend_scene = fiber.preview_render_dict_from_fibers(fibers)
    print(rend_scene)

    img = mi.render(mi.load_dict(rend_scene))
    plt.imshow(img)
    plt.show()
