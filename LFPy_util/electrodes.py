import LFPy_util
import numpy as np

def _planeElectrodes(norm, x, y):
    norm = np.asarray(norm)
    norm = norm/np.linalg.norm(norm)

    X,Y = np.meshgrid(x,y)
    Z = np.zeros(len(X))

    points = np.zeros([3,len(X)])
    points[1,:] = X
    points[2,:] = Y
    points[3,:] = Z

    dx = norm[0]
    dy = norm[1]
    dz = norm[2]

    rot_x = np.arctan2(np.sqrt(dx*dx+dz*dz),dy) + np.pi/2
    rot_y = np.arctan2(dx,dz)

    Rx = LFPy_util.rotation.rotation_matrix([1,0,0],rot_x)
    Ry = LFPy_util.rotation.rotation_matrix([0,1,0],rot_y)
    points = np.dot(Rx,points)
    points = np.dot(Ry,points)

    electrode = {
        'x' : points[0,:],  
        'y' : points[1,:],  
        'z' : points[2,:],  
    }
    return electrode, X, Y

def circularElectrodesXZ(r, n, n_theta, r_0=0,x_0=[0,0,0]):
    """
    Creates a dictionary with positions placed circle in the XZ plane.
    Electrodes will be placed radially away from **x_0** but not closer than **r_0**.
    Theta is varied from :math:`0` to :math:`2\pi` in **n_theta** increments. 

    :param float r:
        Distance to furthest electrode.
    :param int n:
        Number of electrodes between **r_0** and **r**.
    :param int n_theta: 
        Number of directions from :math:`0` to :math:`2\pi`.
    :param float r_0:
        Minimum distance to elctrode from **x_0**.
    :param `~numpy.ndarray` x_0: 
        Origin.
    :returns: 
        *  
         :class:`dict` -- Dictionary with 'x', 'y', 'z' set. 

    Example:
        .. code-block:: python

            # Create circular electrodes.
            electrode_dict = LFPy_util.electrodes.circularElectrodesXZ(
                    r = 200,
                    n = 10,
                    n_theta = 10,
                    r_0 = 20
            ) 
            electrode_dict['sigma'] = 0.3
            # Record the LFP of the electrodes. 
            electrode = LFPy.RecExtElectrode(cell, **electrode_dict)
            electrode.calc_lfp()
    """
    points = np.zeros([3,n_theta*n])
    x_0 = np.array(x_0)
    theta = np.linspace(0,2*np.pi,n_theta,endpoint=False)
    radii = np.linspace(r_0,r,n)
    # Create a disk of points in the xz plane.
    cnt = 0
    for b in theta:
        for a in radii:
            points[0,cnt] = a*np.cos(b) + x_0[0]
            points[1,cnt] = x_0[1]
            points[2,cnt] = a*np.sin(b) + x_0[2]
            cnt += 1
    electrode = {
        'x' : points[0,:],  
        'y' : points[1,:],  
        'z' : points[2,:],  
    }
    return electrode

def circularElectrodesYZ(r, n, n_theta, r_0=0,x_0=[0,0,0]):
    """
    Creates a dictionary with positions placed circle in the XY plane.
    Electrodes will be placed radially away from **x_0** but not closer than **r_0**.
    Theta is varied from :math:`0` to :math:`2\pi` in **n_theta** increments. 

    :param float r:
        Distance to furthest electrode.
    :param int n:
        Number of electrodes between **r_0** and **r**.
    :param int n_theta: 
        Number of directions from :math:`0` to :math:`2\pi`.
    :param float r_0:
        Minimum distance to elctrode from **x_0**.
    :param `~numpy.ndarray` x_0: 
        Origin.
    :returns: 
        *  
         :class:`dict` -- Dictionary with 'x', 'y', 'z' set. 

    Example:
        .. code-block:: python

            # Create circular electrodes.
            electrode_dict = LFPy_util.electrodes.circularElectrodesXZ(
                    r = 200,
                    n = 10,
                    n_theta = 10,
                    r_0 = 20
            ) 
            electrode_dict['sigma'] = 0.3
            # Record the LFP of the electrodes. 
            electrode = LFPy.RecExtElectrode(cell, **electrode_dict)
            electrode.calc_lfp()
    """
    points = np.zeros([3,n_theta*n])
    x_0 = np.array(x_0)
    theta = np.linspace(0,2*np.pi,n_theta,endpoint=False)
    radii = np.linspace(r_0,r,n)
    # Create a disk of points in the xy plane.
    cnt = 0
    for b in theta:
        for a in radii:
            points[0,cnt] = x_0[0]
            points[1,cnt] = a*np.cos(b) + x_0[1]
            points[2,cnt] = a*np.sin(b) + x_0[2]
            cnt += 1
    electrode = {
        'x' : points[0,:],  
        'y' : points[1,:],  
        'z' : points[2,:],  
    }
    return electrode

def circularElectrodesXY(r, n, n_theta, r_0=0,x_0=[0,0,0]):
    """
    Creates a dictionary with positions placed circle in the XY plane.
    Electrodes will be placed radially away from **x_0** but not closer than **r_0**.
    Theta is varied from :math:`0` to :math:`2\pi` in **n_theta** increments. 

    :param float r:
        Distance to furthest electrode.
    :param int n:
        Number of electrodes between **r_0** and **r**.
    :param int n_theta: 
        Number of directions from :math:`0` to :math:`2\pi`.
    :param float r_0:
        Minimum distance to elctrode from **x_0**.
    :param `~numpy.ndarray` x_0: 
        Origin.
    :returns: 
        *  
         :class:`dict` -- Dictionary with 'x', 'y', 'z' set. 

    Example:
        .. code-block:: python

            # Create circular electrodes.
            electrode_dict = LFPy_util.electrodes.circularElectrodesXZ(
                    r = 200,
                    n = 10,
                    n_theta = 10,
                    r_0 = 20
            ) 
            electrode_dict['sigma'] = 0.3
            # Record the LFP of the electrodes. 
            electrode = LFPy.RecExtElectrode(cell, **electrode_dict)
            electrode.calc_lfp()
    """
    points = np.zeros([3,n_theta*n])
    x_0 = np.array(x_0)
    theta = np.linspace(0,2*np.pi,n_theta,endpoint=False)
    radii = np.linspace(r_0,r,n)
    # Create a disk of points in the xy plane.
    cnt = 0
    for b in theta:
        for a in radii:
            points[0,cnt] = a*np.cos(b) + x_0[0]
            points[1,cnt] = a*np.sin(b) + x_0[1]
            points[2,cnt] = x_0[2]
            cnt += 1
    electrode = {
        'x' : points[0,:],  
        'y' : points[1,:],  
        'z' : points[2,:],  
    }
    return electrode


def circularElectrodes(norm, r, n, n_theta, r_0 = 0, x_0=[0,0,0]) :
    """
    Place electrodes on a plane normal to **norm**. The electrodes
    will be placed radially away from **x_0** but not closer than **r_0**.
    Theta is varied from :math:`0` to :math:`2\pi` in **n_theta** increments. 

    The returned values  **x** and **y** are the positions
    of the electrodes in the plane defined in 
    :meth:`~LFPy_util.electrodes.circularElectrodes`
    and relative to the point **r_0**.

    :param norm: Vector normal to the plane.
    :type norm: :class:`~numpy.ndarray` (x,y,z)
    :param float r: Distance to furthest electrode.
    :param int n: Number of electrodes between **r_0** and **r**.
    :param int n_theta: Number of directions from :math:`0` to :math:`2\pi`.
    :param int r_0: Minimum distance to elctrode from **x_0**.
    :param x_0: Origin of **norm**.
    :type x_0: :class:`~numpy.ndarray` (x,y,z)
    :returns: (electrode, x, y) : x and y shape = (nElectrodes)
    :rtype: (dict, :class:`~numpy.ndarray`, :class:`~numpy.ndarray`)
    """
    norm = np.asarray(norm)
    norm = norm/np.linalg.norm(norm)

    points = np.zeros([3,n_theta*n])
    theta = np.linspace(0,2*np.pi,n_theta,endpoint=False)
    radii = np.linspace(r_0,r,n)

    # Create a disk of points in the xy plane.
    cnt = 0
    for b in theta:
        for a in radii:
            points[0,cnt] = a*np.cos(b)
            points[1,cnt] = a*np.sin(b)
            # z coordinate is zero already.
            # points[2,cnt] = 0
            cnt += 1

    # Save the original layout for plotting purposes. 
    x = points[0,:]
    y = points[1,:]

    dx = norm[0]
    dy = norm[1]
    dz = norm[2]

    rot_x = np.arctan2(np.sqrt(dx*dx+dz*dz),dy) + np.pi/2
    rot_y = np.arctan2(dx,dz)

    Rx = LFPy_util.rotation.rotation_matrix([1,0,0],rot_x)
    Ry = LFPy_util.rotation.rotation_matrix([0,1,0],rot_y)
    points = np.dot(Rx,points)
    points = np.dot(Ry,points)

    electrode = {
        'x' : points[0,:],  
        'y' : points[1,:],  
        'z' : points[2,:],  
    }
    return electrode, x, y

def directionElectrodes(x0,x1,n):
    """
    Creates a dictionary with electrodes positioned on a line from 
    x0 to x1. 
    All other parameters needs to be set. 

    :param x0: Inital position
    :type x0: :class:`~numpy.ndarray`: (x,y,z)
    :param x1: End position. 
    :type x1: :class:`~numpy.ndarray` : (x,y,z)
    :param int n: Number of electrodes.
    :returns: Dictionary with 'x', 'y', 'z' set.
    :rtype: dict
    """
    X = np.linspace(x0[0],x1[0],n)
    Y = np.linspace(x0[1],x1[1],n)
    Z = np.linspace(x0[2],x1[2],n)
    grid_electrode_parameters = {
        'x' : X,  
        'y' : Y,
        'z' : Z
    }
    return grid_electrode_parameters

def gridElectrodes(x,y,z):
    """
    Creates a dictionary with positions of electrodes in a grid. The parameters
    are suggested to be generated with :func:`~numpy.linspace`.
    All other parameters needs to be set. 

    :param x,y,z: Vector of points along each axis. 
    :type x,y,z: :class:`~numpy.ndarray`
    :returns: Dictionary with 'x', 'y', 'z' set.
    :rtype: dict

    Example:

    .. code-block:: python

        nx = 10
        ny = 10
        X = np.linspace(-300,300,nx)
        Y = np.linspace(-300,1300,ny)
        Z = [0]
        electrode_dict = electrodes.gridElectrodes(X,Y,Z)
        electrode_dict['sigma'] = 0.3
        electrode = LFPy.RecExtElectrode(cell, **electrode_dict)
        electrode.calc_lfp()
    """
    X,Y,Z = np.meshgrid(x,y,z)
    # Define electrode parameters
    grid_electrode_parameters = {
        'x' : X.flatten(),  # electrode requires 1d vector of positions
        'y' : Y.flatten(),
        'z' : Z.flatten()
    }
    return grid_electrode_parameters
