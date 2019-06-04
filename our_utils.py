import pymesh
from pyntcloud.io import read_ply
from pyntcloud.io import write_ply
import numpy as np
import pandas as pd
#from plyfile import PlyData, PlyElement
import pywavefront
import sys

# define path manually
#path = '/Users/fabischramm/Documents/ADL4CV/data/data/02747177/1b7d468a27208ee3dad910e221d16b18/models/model_normalized.obj'
#path = '/Users/fabischramm/Documents/ADL4CV/adl4cv/ShapeNet_example/1a04e3eab45ca15dd86060f189eb133/model_small.obj'
#path = '/Users/fabischramm/Documents/ADL4CV/data/data/02747177/8e09a584fe5fa78deb69804478f9c547/models/model_normalized.obj'
#path = '/Users/fabischramm/Documents/ADL4CV/data/data/02747177/2ac7a4d88dfedcfef155d75bbf62b80/models/model_normalized.obj'

# get path to model (internal structure in data folder) from call by shell script
path_model = sys.argv[1]
#print(path_model)
path = '/Users/fabischramm/Documents/ADL4CV/data/' + path_model + '/models/model_normalized.obj'
#print(path)

def triangle_area_multi(v1, v2, v3):
    return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)

def extract_color(scene):
    dict={}
    for name, material in scene.materials.items():
        k=0
        for i, v in enumerate(material.vertices):
            if i+k+2>=len(material.vertices):
                break
            dict[(material.vertices[i+k],material.vertices[i+k+1],material.vertices[i+k+2])]=material.diffuse
            k+=2
    return dict
    
def convert_and_sample(path,n=1000, write=True, ret=False):
    
    # 1 - use pymesh
    mesh = pymesh.load_mesh(path)
    points_xyv = mesh.vertices
    v1_xyz = points_xyv[mesh.faces[:,0]]
    v2_xyz = points_xyv[mesh.faces[:,1]]
    v3_xyz = points_xyv[mesh.faces[:,2]]
    
    # 2 - use pywavefront to get vertices and faces
    # to additionally access color information
    mesh2 = pywavefront.Wavefront(path, collect_faces=True)
    points_xyv2 = np.asarray(mesh2.vertices)
    mesh_wavefront = list(mesh2.meshes.values())
    faces = mesh_wavefront[0].faces
    v1_xyz_index = [col[0] for col in faces]
    v1_xyz2 = points_xyv2[v1_xyz_index]
    v2_xyz_index = [col[1] for col in faces]
    v2_xyz2 = points_xyv2[v2_xyz_index]
    v3_xyz_index = [col[2] for col in faces]
    v3_xyz2 = points_xyv2[v3_xyz_index]
    
    # get color info
    mesh_colors=extract_color(mesh2)
    #v1_rgb = mesh_colors[tuple(v3_xyz2[0].tolist())]
    v1_rgb = np.asarray([mesh_colors[tuple(x.tolist())][0:3] for x in v1_xyz2])
    v2_rgb = np.asarray([mesh_colors[tuple(x.tolist())][0:3] for x in v2_xyz2])
    v3_rgb = np.asarray([mesh_colors[tuple(x.tolist())][0:3] for x in v3_xyz2])
    
    # test if two different ways are similar
    #print(points_xyv==points_xyv2)
    #print(v1_xyz== v1_xyz2)
    #print(v2_xyz== v2_xyz2)
    #print(v3_xyz== v3_xyz2)
    
    # use pywavefront to sample, comment out if you want to use pymesh
    points_xyv=points_xyv2
    v1_xyz=v1_xyz2
    v2_xyz=v2_xyz2
    v3_xyz=v3_xyz2
    
    
    areas = triangle_area_multi(v1_xyz, v2_xyz, v3_xyz)
    prob = areas / areas.sum()
    weighted_ind = np.random.choice(range(len(areas)), size=n,p=prob)
    ind = weighted_ind
    v1_xyz = v1_xyz[ind]
    v2_xyz = v2_xyz[ind]
    v3_xyz = v3_xyz[ind]

    v1_rgb = v1_rgb[ind]*255
    v2_rgb = v2_rgb[ind]*255
    v3_rgb = v3_rgb[ind]*255

    u = np.random.rand(n , 1)
    v = np.random.rand(n , 1)
    is_problem = u + v >1
    u[is_problem] = 1 - u[is_problem]
    v[is_problem] = 1 - v[is_problem]
    w = 1 - (u + v)
    result = pd.DataFrame()
    result_xyz = (v1_xyz * u) + (v2_xyz * v) + (v3_xyz * w) 
    result["x"] = result_xyz[:,0]
    result["y"] = result_xyz[:,1]
    result["z"] = result_xyz[:,2]
    
    result_rgb = (v1_rgb * u) + (v2_rgb * v) + (v3_rgb * w)
    result_rgb = result_rgb.astype(np.uint8)
    
    result["red"] = result_rgb[:,0]
    result["green"] = result_rgb[:,1]
    result["blue"] = result_rgb[:,2]
    
    path=path.replace('.obj','.ply')
    if write: #write file
        write_ply(path,points=result,as_text=True)
    if ret:  
        return result

# call the function when using shell_script to process whole folder
# if jupyter notebook is used uncomment this line
convert_and_sample(path, n=50000)