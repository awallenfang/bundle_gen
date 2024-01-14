import numpy as np

def test_intensity_gradiant_phi(theta_size, phi_size):
    phi = np.linspace(0.,1.,theta_size).reshape(theta_size,1)
    intensities = np.ones((theta_size, phi_size)) * phi
    return intensities

def test_intensity_gradiant_theta(theta_size, phi_size):
    theta = np.linspace(0.,1.,phi_size).reshape(1,phi_size)
    intensities = np.ones((theta_size, phi_size)) * theta
    return intensities

def test_intensity_grid(theta_size, phi_size):
    intensities = np.zeros((theta_size, phi_size))
    intensities[0:50,:] = 1
    intensities[100:150,:] = 1
    intensities[200:250,:] = 1
    intensities[300:350,:] = 1
    intensities[400:450,:] = 1

    intensities[:,0:50] = 1
    intensities[:,100:150] = 1
    intensities[:,200:250] = 1
    intensities[:,300:350] = 1
    intensities[:,400:450] = 1
    intensities[:,500:550] = 1
    intensities[:,600:650] = 1
    intensities[:,700:750] = 1
    intensities[:,800:850] = 1
    return intensities
