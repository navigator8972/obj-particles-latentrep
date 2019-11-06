'''
Created on November 26, 2017
@author: optas
'''

import numpy as np
from numpy.linalg import norm
import matplotlib.pylab  as plt
from mpl_toolkits.mplot3d import Axes3D


def rand_rotation_matrix(deflection=1.0, seed=None):
    '''Creates a random rotation matrix.
    deflection: the magnitude of the rotation. For 0, no rotation; for 1, completely random
    rotation. Small deflection => small perturbation.
    DOI: http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
         http://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html
    '''
    if seed is not None:
        np.random.seed(seed)

    randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi    # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi     # For direction of pole deflection.
    z = z * 2.0 * deflection    # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z))

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


def iterate_in_chunks(l, n):
    '''Yield successive 'n'-sized chunks from iterable 'l'.
    Note: last chunk will be smaller than l if n doesn't divide l perfectly.
    '''
    for i in range(0, len(l), n):
        yield l[i:i + n]

        
def add_gaussian_noise_to_pcloud(pcloud, mu=0, sigma=1):
    gnoise = np.random.normal(mu, sigma, pcloud.shape[0])
    gnoise = np.tile(gnoise, (3, 1)).T
    pcloud += gnoise
    return pcloud


def apply_augmentations(batch, conf):
    if conf.gauss_augment is not None or conf.z_rotate:
        batch = batch.copy()

    if conf.gauss_augment is not None:
        mu = conf.gauss_augment['mu']
        sigma = conf.gauss_augment['sigma']
        batch += np.random.normal(mu, sigma, batch.shape)

    if conf.z_rotate:
        r_rotation = rand_rotation_matrix()
        r_rotation[0, 2] = 0
        r_rotation[2, 0] = 0
        r_rotation[1, 2] = 0
        r_rotation[2, 1] = 0
        r_rotation[2, 2] = 1
        batch = batch.dot(r_rotation)
    return batch


def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    '''Returns the center coordinates of each cell of a 3D grid with resolution^3 cells,
    that is placed in the unit-cube.
    If clip_sphere it True it drops the "corner" cells that lie outside the unit-sphere.
    '''
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[norm(grid, axis=1) <= 0.5]

    return grid, spacing

def plot_3d_point_cloud(x, y, z, show=True, show_axis=True, in_u_sphere=False, marker='.', s=8, alpha=.8, figsize=(5, 5), elev=10, azim=240, axis=None, title=None, *args, **kwargs):

    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')        
    else:
        ax = axis
        fig = axis

    if title is not None:
        plt.title(title)

    sc = ax.scatter(x, y, z, marker=marker, s=s, alpha=alpha, *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)

    if in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
    else:
        miv = 1.2 * np.min([np.min(x), np.min(y), np.min(z)])  # Multiply with 0.7 to squeeze free-space. why? change to 1.2
        mav = 1.2 * np.max([np.max(x), np.max(y), np.max(z)])
        ax.set_xlim(miv, mav)
        ax.set_ylim(miv, mav)
        ax.set_zlim(miv, mav)
        plt.tight_layout()

    if not show_axis:
        plt.axis('off')

    if 'c' in kwargs:
        plt.colorbar(sc)

    if show:
        plt.show()

    return fig

def plot_3d_point_cloud_graph(nodes, adjacency=None, axis=None, 
    show=True, show_axis=True, in_u_sphere=False, marker='*', s=8, alpha=.8, figsize=(5, 5), elev=10, azim=240, title=None, *args, **kwargs):
    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')        
        ax.view_init(elev=elev, azim=azim)

        if in_u_sphere:
            ax.set_xlim3d(-0.5, 0.5)
            ax.set_ylim3d(-0.5, 0.5)
            ax.set_zlim3d(-0.5, 0.5)
        else:
            miv = 1.2 * np.min([np.min(nodes[0, :]), np.min(nodes[1, :]), np.min(nodes[2, :])])  # Multiply with 0.7 to squeeze free-space. why? change to 1.2
            mav = 1.2 * np.max([np.max(nodes[0, :]), np.max(nodes[1, :]), np.max(nodes[2, :])])
            ax.set_xlim(miv, mav)
            ax.set_ylim(miv, mav)
            ax.set_zlim(miv, mav)
            plt.tight_layout()

        if not show_axis:
            plt.axis('off')
    else:
        ax = axis
    
    sc = ax.scatter(nodes[0, :], nodes[1, :], nodes[2, :], marker=marker, s=s, alpha=alpha)

    if adjacency is not None:
        assert(adjacency.shape[0] == adjacency.shape[1] and adjacency.shape[0]==nodes.shape[1])
        for i in range(nodes.shape[0]):
            for j in range(nodes.shape[1]):
                ax.plot([nodes[0][i], nodes[0][j]], [nodes[1][i], nodes[1][j]], zs=[nodes[2][i], nodes[2][j]], color='k', alpha=adjacency[i][j])

    if 'c' in kwargs:
        plt.colorbar(sc)    
    # if show:
    #     plt.show()

    return ax

import torch

# def rotmatrix_from_axisangle(axisangle):
#     '''
#     axisangle: (batch_size, 4)  with [:, :3] representing axes while [:, -1] representing angle radian

#     return:     (batch_size, 3, 3) rotation matrices 
#     '''
#     #normalize the axes
#     axisangle[:, :3] = axisangle[:, :3] / axisangle[:, :3].norm(dim=1)
#     rot_mat = torch.zeros((axisangle.shape[0], 3, 3))
#     cos_theta = torch.cos(axisangle[:, -1])
#     sin_theta = torch.sin(axisangle[:, -1])
#     rot_mat[:, 0, 0] = cos_theta + axisangle[:, 0]**2*(1-cos_theta)
    
#     return

def batch_rodriguez_formula(pnts, axisangle):
    '''
    Rotate given batched points according to axisangle
    pnts:       (batch_size_p, 3, n)
    axisangle:  (batch_size_r, 4)  with [:, :3] representing axes while [:, -1] representing angle radian

    return:     (batch_size_p, batch_size_r, 3, n) rotated points 
    '''

    batch_p, n_pnts = pnts.shape[0], pnts.shape[2]
    batch_r = axisangle.shape[0]
    #normalize the axes
    axes = torch.nn.functional.normalize(axisangle[:, :3], dim=1)
    axes = axes[None, :, None, :].repeat(batch_p, 1, n_pnts, 1)                                 #(batch_p, batch_r, n_pnts, 3)

    #v_rot = v cos_theta + (k x v) sin_theta + k (k \dot v)(1-cos_theta)
    pnts_expand = pnts[:, None, :, :].repeat(1, batch_r, 1, 1).transpose(2, 3).contiguous()     #(batch_p, batch_r, n, 3)
    
    theta = axisangle[:, -1][None, :, None, None].repeat(batch_p, 1, 1, 1)
    cos_theta = torch.cos(theta)                                                                 #(batch_p, batch_r, 1, 1)
    sin_theta = torch.sin(theta)                                                                 #(batch_p, batch_r, 1, 1)

    v_cos_theta = pnts_expand * cos_theta                                                       #(batch_p, batch_r, n, 3)

    k_x_v_sin_theta = torch.cross(axes, pnts_expand, dim=3)                                     #(batch_p, batch_r, n, 3)
    k_x_v_sin_theta = k_x_v_sin_theta * sin_theta
    k_kdotv = axes * ( torch.bmm(axes.view(-1, 3).unsqueeze(1), pnts_expand.view(-1, 3).unsqueeze(2)).view(batch_p, batch_r, n_pnts, 1) )     #(batch_p, batch_r, n, 3) * (batch_p, batch_r, )

    ret = v_cos_theta + k_x_v_sin_theta + k_kdotv * (1-cos_theta)
    return ret.transpose(2, 3).contiguous()

def batch_rodriguez_formula_elementwise(pnts, axisangle):
    '''
    Rotate given batched points according to axisangle, do an elementwise operation within the batch
    pnts:       (batch_size_p, 3, n)
    axisangle:  (batch_size_p, 4)  with [:, :3] representing axes while [:, -1] representing angle radian, batch must contain batch_size_p elements

    return:     (batch_size_p, 3, n) rotated points 
    '''
    batch_p, n_pnts = pnts.shape[0], pnts.shape[2]
    axes = torch.nn.functional.normalize(axisangle[:, :3], dim=1)                               #(batch_p, 3)
    axes = axes[:, None, :].repeat(1, n_pnts, 1)                                                #(batch_p, n_pnts, 3)

    #v_rot = v cos_theta + (k x v) sin_theta + k (k \dot v)(1-cos_theta)
    pnts_expand = pnts.transpose(1, 2).contiguous()                                             #(batch_p, n_pnts, 3)
    
    theta = axisangle[:, -1][:, None, None]
    cos_theta = torch.cos(theta)                                                                #(batch_p, 1, 1)
    sin_theta = torch.sin(theta)                                                                #(batch_p, 1, 1)

    v_cos_theta = pnts_expand * cos_theta                                                       #(batch_p, n, 3)

    k_x_v_sin_theta = torch.cross(axes, pnts_expand, dim=2)                                     #(batch_p, n, 3)
    k_x_v_sin_theta = k_x_v_sin_theta * sin_theta
    k_kdotv = axes * ( torch.bmm(axes.view(-1, 3).unsqueeze(1), pnts_expand.view(-1, 3).unsqueeze(2)).view(batch_p, n_pnts, 1))    #(batch_p, n, 1)

    ret = v_cos_theta + k_x_v_sin_theta + k_kdotv * (1-cos_theta)
    return ret.transpose(1, 2).contiguous()


if __name__ == '__main__':
    #prepare a test
    import tf.transformations as tf_trans

    pnts = np.random.rand(5, 3, 10)
    axisangle = np.random.rand(5, 4)
    axisangle[:, :3] = axisangle[:, :3] / np.linalg.norm(axisangle[:, :3],axis=1)[:, None]

    numpy_ret = np.zeros((pnts.shape[0], axisangle.shape[0], pnts.shape[1], pnts.shape[2]))

    for i in range(axisangle.shape[0]):
        rot = tf_trans.rotation_matrix(axisangle[i][-1], axisangle[i][:3])[:3, :3]
        for j in range(pnts.shape[0]):
            numpy_ret[j, i] = rot.dot(pnts[j])
    
    torch_ret = batch_rodriguez_formula(torch.from_numpy(pnts), torch.from_numpy(axisangle)).detach().numpy()
    torch_ret_elementwise = batch_rodriguez_formula_elementwise(torch.from_numpy(pnts), torch.from_numpy(axisangle)).detach().numpy()
    print(torch_ret[0][0])
    print(torch_ret_elementwise[0])
    print(numpy_ret[0][0])
    
    print(np.max(np.abs(numpy_ret-torch_ret)))
    print(np.max(np.abs(np.moveaxis(np.diagonal(numpy_ret), -1, 0)-torch_ret_elementwise)))