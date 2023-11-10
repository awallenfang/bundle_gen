import mitsuba as mi
import drjit as dr
import numpy as np

import matplotlib.pyplot as plt

import fiber
from brdf import TabulatedBCRDF, cartesian_to_polar

def cartesian_to_polar(x,y,z):
    phi = np.arctan2(y,x)
    theta = np.arccos(z / np.sqrt(x*x + y*y + z*z))

    return (theta, phi)


# length = r/ (sqrt(1 - dot(out, fiber_dir)^2))
# wenn dot nahe 1 wegwerfen

# Check code where exactly it's written if it's even uniform
# Get the test cases to work correctly

TEST = False

RAY_AMT = 100000
OUT_PHI = 100
OUT_THETA = 100

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
    scene: mi.Scene = mi.load_dict(fiber.scene_dict_from_fibers(fibers))

    sampler: mi.Sampler = mi.load_dict({'type': 'independent'})
    # sampler.seed(int(time.time()), RAY_AMT)    
    sampler.seed(231, RAY_AMT)    

    directions = mi.Vector3f(1,0,0)
    origins = mi.Point3f(-20,0,0)
    magnitudes = mi.Float(1.)

    bounce_n = mi.UInt32(0)
    max_bounce = mi.UInt32(200)
    active: mi.Mask = mi.Mask(True)

    brdf = TabulatedBCRDF(["fiber_0/fiber_0_lambda" + str(i) + "_TM_depth6.binary" for i in range(24)])

    loop = mi.Loop("Tracing", lambda: (active, directions, origins, magnitudes,  bounce_n, max_bounce))

    while loop(active):
        ray = mi.Ray3f(origins, directions)
        intersection: mi.SurfaceInteraction3f = scene.ray_intersect(ray, active=active)

        # Check if the ray is valid before any brdfs are run
        t_too_big = mi.Mask(intersection.t > 999999999)
        t_too_small = mi.Mask(intersection.t < 0)
        active &= ~t_too_big
        active &= ~t_too_small

        # dr.printf_async("Ori: (%f,%f,%f)\n", origins.x, origins.y, origins.z)
        # dr.printf_async("Dir: (%f,%f,%f)\n", directions.x, directions.y, directions.z)
        # dr.printf_async("t: %f\n", intersection.t)

        output: mi.Shape = intersection.shape
        new_ori, new_dir, new_mag = brdf.brdf(intersection, active, sampler, 414.)

        origins[active] = new_ori
        directions[active] = new_dir
        magnitudes[active] *= new_mag
        bounce_n[active] += 1
        active &= bounce_n < max_bounce
        active &= magnitudes <= 0.000001
    print(dr.max(bounce_n))
    out_model = np.zeros((OUT_THETA, OUT_PHI))

    n_dir = directions.numpy()
    n_mag = magnitudes.numpy()

    phis = np.arctan2(n_dir[:,1], n_dir[:,0])
    thetas = np.arccos(n_dir[:,2] / np.linalg.norm(n_dir, axis=1))

    phis[phis<0] += np.pi*2
    phi_coords = np.floor((phis / (2*np.pi)) * OUT_PHI)
    phi_coords = phi_coords.astype(int)
    print(np.max(phi_coords))

    theta_coords = np.floor((thetas / np.pi) * OUT_THETA)
    theta_coords = theta_coords.astype(int)
    # print(np.max(theta_coords))

    coords = np.stack((theta_coords, phi_coords), axis=1)
    # print(coords)
    # print(n_mag)

    
    # print(f'Coords: {coords}\n')
    # coords_sorter = np.argsort(coords[:,0], axis=0)
    # print(f'Sorter: {coords_sorter}\n')
    # sorted_coords = coords[coords_sorter]
    # sorted_mags = n_mag[coords_sorter]
    # print(f'Sorted Coords: {sorted_coords}\n')

    
    out_model[theta_coords, phi_coords] += n_mag
    
    plt.imshow(out_model)
    plt.show()


    # print(thetas)
    # # numpy scatter
    # for i in range(0, RAY_AMT):
    #     dir = n_dir[i]
    #     theta, phi = cartesian_to_polar(dir[0], dir[1], dir[2])
    #     pos_x = int(np.floor(((phi + 2*np.pi * (phi < 0)) / (np.pi*2)) * OUT_PHI))
    #     pos_y = int(np.floor((theta / (np.pi)) * OUT_THETA))
        
    #     mag = n_mag[i]

    #     out_model[pos_y, pos_x] += mag / np.sin(theta)

    # fig, ax = plt.subplots(1)
    # ax.set_xlabel("phi")
    # ax.set_ylabel("theta")
    # ax.imshow(out_model)
    # plt.show()

else:
    rend_scene = fiber.preview_render_dict_from_fibers(fibers)
    print(rend_scene)

    img = mi.render(mi.load_dict(rend_scene))
    plt.imshow(img)
    plt.show()
