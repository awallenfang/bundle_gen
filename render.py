import time

import mitsuba as mi
import drjit as dr

from fiber import scene_dict_from_fibers, get_bounds
from util import mitsuba_cartesian_to_polar

class Renderer():
    def __init__(self, in_fibers, brdf, samples=100000, seed=-1, bounces=100, in_dir=mi.Vector3f(1.,0.,0.), in_pos=mi.Point3f(-20.,0.,0.), spread_amt=1, out_size_phi=100, out_size_theta=100):
        if seed == -1:
            self.seed = int(time.time())
        else: 
            self.seed = seed

        self.ray_amt = samples
        self.scene = mi.load_dict(scene_dict_from_fibers(in_fibers))
        self.radius, self.center_x, self.center_y = get_bounds(in_fibers)
        self.brdf = brdf
        self.bounces = bounces
        self.origin = in_pos
        self.in_direction = in_dir
        self.spread_amt = spread_amt
        self.out_size_phi = out_size_phi
        self.out_size_theta = out_size_theta

    def render_structure(self):
        """
        Render the fiber structure that was set up
        """
        print(f'Running render with following settings:\n\nSeed: {self.seed}\nInitial Origin: {self.origin}\nInitial Direction: {self.in_direction}\nMax Bounces: {self.bounces}\nAmount of rays: {self.ray_amt}\n')
        # Set up the sampler
        sampler: mi.Sampler = mi.load_dict({'type': 'independent'})
        sampler.set_sample_count(self.bounces * self.ray_amt)
        sampler.set_samples_per_wavefront(self.bounces)    
        sampler.seed(self.seed, self.ray_amt)
        
        up = mi.Vector3f(0.,0.,1.)

        origins = self.origin
        
        if self.spread_amt > 1:
            side = dr.normalize(dr.cross(self.in_direction, up)) * self.radius
            rand_offset = sampler.next_1d()
            origins = dr.lerp(origins + side, origins - side,  rand_offset)
            

        # Set up the running variables
        directions = self.in_direction
        magnitudes = mi.Float(1.)

        bounce_n = mi.UInt32(0)
        max_bounce = mi.UInt32(self.bounces)
        active: mi.Mask = mi.Mask(True)

        n_too_big = mi.UInt32(0)
        n_too_small = mi.UInt32(0)

        # Start the loop, which runs until no ray is active anymore
        loop = mi.Loop("Tracing", lambda: (active, directions, origins, magnitudes,  bounce_n, max_bounce, n_too_big, n_too_small, sampler))

        while loop(active):
            # Create the ray and intersect the scene
            ray = mi.Ray3f(origins, directions)
            intersection: mi.SurfaceInteraction3f = self.scene.ray_intersect(ray, active=active)

            # Check if the ray is valid before any brdfs are run
            t_too_big = mi.Mask(intersection.t > 999999999999999999)
            t_too_small = mi.Mask(intersection.t < 0)

            active &= ~t_too_big
            n_too_big[t_too_big] = 1
            active &= ~t_too_small
            n_too_small[t_too_small] = 1

            # dr.printf_async("Ori: (%f,%f,%f)\n", origins.x, origins.y, origins.z)
            # dr.printf_async("Dir: (%f,%f,%f)\n", directions.x, directions.y, directions.z)
            # dr.printf_async("t: %f\n\n", intersection.t)
            
            rand_2d = sampler.next_2d()
            new_dir = mi.warp.square_to_uniform_sphere(rand_2d)
            # sampler.advance()


            new_ori, _, new_mag = self.brdf.brdf(intersection, new_dir, sampler, 600.)

            # Update the running variables
            origins[active] = new_ori
            directions[active] = new_dir
            magnitudes[active] *= new_mag
            bounce_n[active] += 1

            # Update the active mask
            active &= bounce_n < max_bounce
            # active &= magnitudes <= 0.00000000000000000000000001
        print(f'\nResults:\n\nMaximum bounce depth: {dr.max(bounce_n)[0]}\nMaximum vertical offset: {dr.max(dr.abs(origins.z))[0]}\nBounce stopped because ray left the structure: {dr.sum(n_too_big)[0]}')

        
        out_model = dr.empty(mi.Float, self.out_size_phi * self.out_size_theta)

        thetas, phis = mitsuba_cartesian_to_polar(directions)

        x_indices = mi.UInt(dr.floor((phis / (dr.pi * 2.)) * self.out_size_phi))
        y_indices = mi.UInt(dr.floor(((thetas + (dr.pi / 2.)) / dr.pi) * self.out_size_theta))

        out_phi_size = mi.UInt32(self.out_size_phi)
        indices = x_indices + out_phi_size * y_indices

        dr.scatter_reduce(dr.ReduceOp.Add, out_model, magnitudes, indices)
        # Return the directions and magnitudes as numpy vectors
        n_dir = directions.numpy()
        n_mag = magnitudes.numpy()

        return (out_model.numpy())