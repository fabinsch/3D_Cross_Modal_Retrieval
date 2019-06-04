#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 14:52:59 2019

@author: fabischramm
"""
import pickle
import pymesh

mesh = pymesh.load_mesh('ShapeNet_example/1a9b552befd6306cc8f2d5fe7449af61/model.obj')
#mesh = pymesh.load_mesh('ShapeNet_example/1131091-3ef186720b1bb6e704c46aebf57a42677d1638af/cube.obj')

# inspired by https://gist.github.com/ricklentz/7993abd7c6b9d73cf690a270b51c04f7
#pymesh.save_mesh_raw('ShapeNet_example/1a9b552befd6306cc8f2d5fe7449af61/model.obj'.replace(".obj",".ply"), mesh.vertices, mesh.faces, mesh.voxels)

# with pymesh doc https://pymesh.readthedocs.io/en/latest/basic.html

# check the dimensions
print(mesh.num_vertices, mesh.num_faces, mesh.num_voxels)
print(mesh.dim, mesh.vertex_per_face, mesh.vertex_per_voxel)
print(mesh.get_attribute_names())
# different ways of saving to pyl
#pymesh.save_mesh('ShapeNet_example/1a9b552befd6306cc8f2d5fe7449af61/test1.ply', mesh, use_float=True, ascii=True)
#pymesh.save_mesh_raw('ShapeNet_example/1a9b552befd6306cc8f2d5fe7449af61/test2.ply', mesh.vertices, mesh.faces, use_float=True, ascii=True)
pymesh.save_mesh('ShapeNet_example/1a04e3eab45ca15dd86060f189eb133/test1.ply', mesh, use_float=True, ascii=True)

corner_texture=mesh.get_attribute('corner_texture');
for i in range(len(corner_texture)):
    print(corner_texture[i])