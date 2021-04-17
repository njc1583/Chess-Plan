import time
from klampt import *
from klampt import vis
from klampt.math import vectorops,so3,se3
from klampt.model.trajectory import Trajectory,SE3Trajectory
from klampt.model import ik
from klampt.io import numpy_convert
import numpy as np
import math
import random
import sys
sys.path.append('../common')
import gripper
import known_grippers
import normals


from klampt.model.contact import ContactPoint

#global data structure, will be filled in for you
object_normals = None

class AntipodalGrasp:
    """A structure containing information about antipodal grasps.
    
    Attributes:
        center (3-vector): the center of the fingers (object coordinates).
        axis (3-vector): the direction of the line through the
            fingers (object coordinates).
        approach (3-vector, optional): the direction that the fingers
            should move forward to acquire the grasp.
        finger_width (float, optional): the width that the gripper should
            open between the fingers.
        contact1 (ContactPoint, optional): a point of contact on the
            object.
        contact2 (ContactPoint, optional): another point of contact on the
            object.
    """
    def __init__(self,center,axis):
        self.center = center
        self.axis = axis
        self.approach = None
        self.finger_width = None
        self.contact1 = None
        self.contact2 = None

    def add_to_vis(self,name,color=(1,0,0,1)):
        finger_radius = 0.02
        if self.finger_width == None:
            w = 0.05
        else:
            w = self.finger_width*0.5+finger_radius
        a = vectorops.madd(self.center,self.axis,w)
        b = vectorops.madd(self.center,self.axis,-w)
        vis.add(name,[a,b],color=color)
        if self.approach is not None:
            vis.add(name+"_approach",[self.center,vectorops.madd(self.center,self.approach,0.05)],color=(1,0.5,0,1))
        normallen = 0.05
        if self.contact1 is not None:
            vis.add(name+"cp1",self.contact1.x,color=(1,1,0,1),size=0.01)
            vis.add(name+"cp1_normal",[self.contact1.x,vectorops.madd(self.contact1.x,self.contact1.n,normallen)],color=(1,1,0,1))
        if self.contact2 is not None:
            vis.add(name+"cp2",self.contact2.x,color=(1,1,0,1),size=0.01)
            vis.add(name+"cp2_normal",[self.contact2.x,vectorops.madd(self.contact2.x,self.contact2.n,normallen)],color=(1,1,0,1))

def grasp_from_contacts(contact1,contact2):
    """Helper: if you have two contacts, this returns an AntipodalGrasp"""
    d = vectorops.unit(vectorops.sub(contact2.x,contact1.x))
    grasp = AntipodalGrasp(vectorops.interpolate(contact1.x,contact2.x,0.5),d)
    grasp.finger_width = vectorops.distance(contact1.x,contact2.x)
    grasp.contact1 = contact1
    grasp.contact2 = contact2
    return grasp

def ray_cast_to_contact_point(obj,source,direction, object_normals):
    """Produces a ContactPoint for the point hit by the ray:
        x(t) = source + t*direction
    
    Assumes object_normals is set up to match the object.
    
    Arguments:
        obj (RigidObjectModel)
        source (3-vector)
        direction (3-vector)
    
    Returns: ContactPoint with `x` set to the hit point,
    `n` set to the triangle normal, and coefficient of friction = 1.
    
    None is returned if the object is not hit by the ray.
    """
    assert object_normals is not None
    hit_tri,pt = obj.geometry().rayCast_ext(source,direction)
    if hit_tri < 0:
        return None
    return ContactPoint(pt,object_normals[hit_tri],1)

def fill_in_grasp(grasp,rigid_object, object_normals):
    """TODO: you should fill this out to generate
    grasp.contact1, grasp.contact2 and grasp.finger_width.  
    
    Arguments:
        grasp (AntipodalGrasp): a partially-specified grasp (center and axis)
        rigid_object (RigidObjectModel): the object
    
    Returns None. (Just fill in the members of the grasp object.)
    """
    multiplier = 1
    if grasp.contact1 is not None and grasp.contact2 is not None:
        return
    grasp.contact1 = ray_cast_to_contact_point(rigid_object, grasp.center, vectorops.mul(grasp.axis,multiplier), object_normals)
    if grasp.contact1 is None: # try ray in other direction
        multiplier = -1
        grasp.contact1 = ray_cast_to_contact_point(rigid_object, grasp.center, vectorops.mul(grasp.axis,multiplier), object_normals)

    if grasp.contact1 is not None:
        # Move in the direction opposite the first contact point
        newCenter = vectorops.add(grasp.contact1.x,vectorops.mul(grasp.axis,50*multiplier))
        #print("newCenter:", newCenter, "Orig center:", grasp.center)
        # Move in the opposite direction to "hit" the object again from another source with raycast
        grasp.contact2 = ray_cast_to_contact_point(rigid_object, newCenter, vectorops.mul(grasp.axis,-1*multiplier), object_normals)
        # Handle case where original grasp center is inside the object(like in can obj)
        if grasp.contact2 != None and vectorops.distance(grasp.contact2.x, grasp.contact1.x)<0.0000001:
            # Go in the other direction from the original center if the two contact points are found to be the same
            grasp.contact2 = ray_cast_to_contact_point(rigid_object, grasp.center, vectorops.mul(grasp.axis,-1*multiplier), object_normals)

    if grasp.contact1 is not None and grasp.contact2 is not None:
        grasp.finger_width = vectorops.distance(grasp.contact1.x,grasp.contact2.x)
        print("Grasp 1:",grasp.contact1.x)
        print("Grasp 2:",grasp.contact2.x)
        print("Finger Width:", grasp.finger_width)
    else:
        print("**No Solution**")
    return

def antipodal_grasp_sample_volume(gripper,rigid_object,k):
    """Samples the top k high quality antipodal grasps for the
    object out of N sampled centers and axes within the object
    
    Returns a list of up to k (grasp,score) pairs, sorted in order
    of increasing score.
    """
    #this is a bounding box for the object
    bmin,bmax = rigid_object.geometry().getBBTight()
    print(bmin,bmax)
    grasps = []
    object_normals = normals.get_object_normals(rigid_object)
    for i in np.linspace(bmin[2], bmax[2], num=50): # iterate through all obj z values
        center = [0,0,i]
        for j in range(10):
            axis = [np.random.uniform(low=-2*np.pi, high=2*np.pi),np.random.uniform(low=-2*np.pi, high=2*np.pi),0]
            axis = vectorops.unit(axis)
            grasp = AntipodalGrasp(center, axis)
            fill_in_grasp(grasp,rigid_object, object_normals)
            if grasp.finger_width == None:
                print("Fail:",grasp.center, grasp.axis, grasp.contact1, grasp.contact2)
            else:
                grasps.append(grasp)
    scores = [(g.finger_width,g) for g in grasps] # sort by finger width, 
    scores = sorted(scores,key = lambda x:x[0], reverse=True)
    if k > len(grasps): k=len(grasps)
    return [(x[1],x[0]) for x in scores[:k]]