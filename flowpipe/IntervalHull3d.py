import numpy as np

def yaw(theta):
    theta = np.deg2rad(theta)
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta),  np.cos(theta), 0],
                     [            0,              0, 1]])

def pitch(theta):
    theta = np.deg2rad(theta)
    return np.array([[np.cos(theta) , 0, np.sin(theta)],
                     [             0, 1,             0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def roll(theta):
    theta = np.deg2rad(theta)
    return np.array([[1,              0,             0],
                     [0,  np.cos(theta), np.sin(theta)],
                     [0, -np.sin(theta), np.cos(theta)]])

def bbox(data):
    means = np.mean(data, axis=1)
    cov = np.cov(data)
    eval, evec = np.linalg.eig(cov)
    centered_data = data - means[:,np.newaxis]
    aligned_coords = np.matmul(evec.T, centered_data)
    xmin, xmax, ymin, ymax, zmin, zmax = np.min(aligned_coords[0, :]), np.max(aligned_coords[0, :]), np.min(aligned_coords[1, :]), np.max(aligned_coords[1, :]), np.min(aligned_coords[2, :]), np.max(aligned_coords[2, :])
    rectCoords = lambda x1, y1, z1, x2, y2, z2: np.array([[x1, x1, x2, x2, x1, x1, x2, x2],
                                                        [y1, y2, y2, y1, y1, y2, y2, y1],
                                                        [z1, z1, z1, z1, z2, z2, z2, z2]])

    realigned_coords = np.matmul(evec, aligned_coords)
    realigned_coords += means[:, np.newaxis]

    rrc = np.matmul(evec, rectCoords(xmin, ymin, zmin, xmax, ymax, zmax))
    rrc += means[:, np.newaxis] 
    return rrc