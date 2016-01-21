import numpy as np
import json
import os
import shutil
from skimage import measure
from neuron import h

def visualizeNeuron(electrodes=[],path='.'):
    """
    Initialize a new directory ready for visualiztion with Three.js and WebGL.

    The new folder will be named "neuron".
    To start the program enter the created directory with a terminal,
    create a simple HTTP file server and open a webpage at 'localhost:8000'.

    Ex. 
        $ python -m 'SimpleHTTPServer'

    And open 'localhost:8000' in a web broswer.

    :param list electrodes: List of electrode dictionaries. 
    :param str path: Relative or full directory address.
    """
    folder_name = '/neuron'
    path = path + folder_name
    if os.path.exists(path):
        shutil.rmtree(path)
    web_template_dir = os.path.dirname(os.path.realpath(__file__))
    web_template_dir = web_template_dir + '/../web_templates/neuron'
    # Copy the web_template folder into the desired directory. 
    shutil.copytree(web_template_dir, path)

    writeNeuronToJson(directory=path)
    writeElectrodesToJson(electrodes,directory=path)

def visualizeNeuronAndPotential(LFP,x,y,z,n=1, electrodes=None, 
        path='.') :
    folder_name = '/neuron_contours'
    path = path + folder_name
    if os.path.exists(path):
        shutil.rmtree(path)
    web_template_dir = os.path.dirname(os.path.realpath(__file__))
    web_template_dir = web_template_dir + '/../web_templates/neuron_contours'
    # Copy the web_template folder into the desired directory. 
    shutil.copytree(web_template_dir, path)

    writeNeuronToJson(directory=path)
    writeContoursToJson(LFP,x,y,z,n=n,directory=path)
    if electrodes != None :
        writeElectrodesToJson(electrodes,directory=path)

def writeElectrodesToJson(electrodes,filename='electrodes',directory='neuron'):
    """
    Create a JSON file of the electrode positions. 

    :param list electrodes: List of electrode dictionaries. 
    :param str filename: Filename.
    :param str directory: Relative or full directory address.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename, ext = os.path.splitext(filename)
    filename = filename + '.js'
    filename = directory + '/' + filename

    if not isinstance(electrodes,list) :
        electrodes = [electrodes]
    if len(electrodes) == 0:
        return

    x = np.asarray([])
    y = np.asarray([])
    z = np.asarray([])

    for item in electrodes:
        x = np.append(x,np.asarray(item["x"]))
        y = np.append(y,np.asarray(item["y"]))
        z = np.append(z,np.asarray(item["z"]))

    data = { 
        'x': x.tolist(), 
        'y': y.tolist(), 
        'z': z.tolist(), 
    }
    dataString = json.dumps(data)
    f = open(filename,'w')
    f.write(dataString)
    f.close()

def writeNeuronToJson(filename='neuron', directory='neuron'):
    """
    Create a JSON file of the morphology of the currently loaded neuron.
    The position of each compartment are stored along with the diameter
    at each of those positions. Each section is seperated with a row of zeros.

    :param str filename: Filename.
    :param str directory: Relative or full directory address.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename, ext = os.path.splitext(filename)
    filename = filename + '.js'
    filename = directory + '/' + filename

    x = [];
    y = [];
    z = [];
    diam = [];

    arrayLength = 0;
    for sec in h.allsec():
        arrayLength += int(h.n3d());
        # Create a gap in the array to seperate sections.
        arrayLength += 1;
    x = np.zeros(arrayLength)
    y = np.zeros(arrayLength)
    z = np.zeros(arrayLength)
    diam = np.zeros(arrayLength)

    indx = 0;
    for sec in h.allsec():
        for i in xrange(int(h.n3d())):
            x[indx] = h.x3d(i);
            y[indx] = h.y3d(i);
            z[indx] = h.z3d(i);
            diam[indx] = h.diam3d(i);
            indx += 1;
        indx += 1;

    x = np.zeros(arrayLength)
    y = np.zeros(arrayLength)
    z = np.zeros(arrayLength)
    diam = np.zeros(arrayLength)
        
    indx = 0;
    for sec in h.allsec():
        for i in xrange(int(h.n3d())):
            x[indx] = h.x3d(i);
            y[indx] = h.y3d(i);
            z[indx] = h.z3d(i);
            diam[indx] = h.diam3d(i);
            indx += 1;
        indx += 1;
    
    if len(diam) == 0:
        print "No sections found."
        return
    data = { \
        'x': x.tolist(), \
        'y': y.tolist(), \
        'z': z.tolist(), \
        'diam': diam.tolist(), \
    }
    dataString = json.dumps(data)
    f = open(filename,'w')
    f.write(dataString)
    f.close()

def writeContoursToJson(LFP,x,y,z,n=1,directory='neuron'):
    dname_base = 'contour_'
    fname_base = 'frame_'
    config_name = 'config.js'
    dirs_config_name = 'contour_dir_list.js'

    dx = np.abs(x[1]-x[0])
    dy = np.abs(y[1]-y[0])
    dz = np.abs(z[1]-z[0])

    nx = len(x)
    ny = len(y)
    nz = len(z)

    baseline = 0
    # Choose n isopotentials between max amplitude and baseline, not 
    # including endpoints.
    amp_max = LFP.max()
    values_pos = []
    if amp_max > 0 : 
        values_pos = np.linspace(baseline,amp_max,n+2)
        values_pos = values_pos[1:-1]

    amp_min = LFP.min()
    values_neg = []
    if amp_min < 0 : 
        values_neg = np.linspace(baseline,amp_min,n+2)
        values_neg = values_neg[1:-1]

    contour_directories = []
    contour_cnt = 0
    for iso_pot_val in values_pos:
        # Create a seperate folder for each contour animation.
        dir_name = dname_base+'%04d' %(contour_cnt)
        contour_directories.append(dir_name)
        dir_path = directory+'/'+dir_name
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        for i in xrange(LFP.shape[1]):
            U = LFP[:,i].reshape(nx,ny,nz)
            # The marching cubes algorithm cannot deal with levels
            # outside the range of U. 
            if U.max() < iso_pot_val or U.min() > iso_pot_val:
                verts = np.zeros([0,3])
                faces = np.zeros([0,3])
            else :
                verts, faces = measure.marching_cubes(
                        U,
                        iso_pot_val,
                        spacing=(dx,dy,dz)
                )
                verts[:,0] = verts[:,0] + x[0]
                verts[:,1] = verts[:,1] + y[0]
                verts[:,2] = verts[:,2] + z[0]
            filename = fname_base + '%04d.js' %(i)
            meshToJson(verts,faces,filename,dir_path)

        # Save a config file listing all the data files. 
        data = {'n_frames' : LFP.shape[1]}
        dataString = json.dumps(data)
        f = open(dir_path+'/'+config_name,'w')
        f.write(dataString)
        f.close()
        contour_cnt += 1
    for iso_pot_val in values_neg:
        # Create a seperate folder for each contour animation.
        dir_name = dname_base+'%04d' %(contour_cnt)
        contour_directories.append(dir_name)
        dir_path = directory+'/'+dir_name
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        for i in xrange(LFP.shape[1]):
            U = LFP[:,i].reshape(nx,ny,nz)
            # The marching cubes algorithm cannot deal with levels
            # outside the range of U. 
            if U.max() < iso_pot_val or U.min() > iso_pot_val:
                verts = np.zeros([0,3])
                faces = np.zeros([0,3])
            else :
                verts, faces = measure.marching_cubes(
                        U,
                        iso_pot_val,
                        spacing=(dx,dy,dz)
                )
                verts[:,0] = verts[:,0] + x[0]
                verts[:,1] = verts[:,1] + y[0]
                verts[:,2] = verts[:,2] + z[0]
            filename = fname_base + '%04d.js' %(i)
            meshToJson(verts,faces,filename,dir_path)

        # Save a config file listing all the data files. 
        data = {'n_frames' : LFP.shape[1]}
        dataString = json.dumps(data)
        f = open(dir_path+'/'+config_name,'w')
        f.write(dataString)
        f.close()
        contour_cnt += 1
    # Save a config file listing all the data files. 
    data = {'n_contours' : len(contour_directories)}
    dataString = json.dumps(data)
    f = open(directory+'/'+dirs_config_name,'w')
    f.write(dataString)
    f.close()

def meshToJson(verts,faces,filename='potential.js',directory='contours') :
    """
    Create a JSON file of the verts and faces 
    compatible with Three.js's JSON loader.

    :param verts: Position touples. 
    :type verts: :class:`~numpy.ndarray` shape = (nVerts,3)
    :param faces: Vertex indices in counter-clockwise order. 
    :type faces: :class:`~numpy.ndarray` shape = (nFaces,3)
    :param str filename: Filename.
    :param str directory: Relative or full directory address.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Remove the extension and add it again to handle filenames with .js ending.
    filename, ext = os.path.splitext(filename)
    path = directory + '/' + filename + '.js'
    N = faces.shape[0]
    # This version and mode of three.js json parser requires that 
    # vertices start with a 0.
    facesFormatted = np.zeros((N,4))
    facesFormatted[:,1:] = faces

    data = { \
        'metadata' : \
            {\
            'formatVersion' : 3\
            },\
        'vertices': verts.flatten().tolist(), \
        'faces': facesFormatted.astype(int).flatten().tolist()\
    }
    dataString = json.dumps(data)
    f = open(path,'w')
    f.write(dataString)
    f.close()



    

