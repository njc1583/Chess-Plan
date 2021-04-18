from klampt import WorldModel,Simulator
from klampt.io import resource
from klampt import vis 
from klampt.model.trajectory import Trajectory,RobotTrajectory, execute_path,path_to_trajectory
from klampt.math import vectorops,so3,se3
import math
import random
import time
import sys
from klampt.model import trajectory
from klampt.robotsim import RigidObjectModel
sys.path.append("../common")
sys.path.append("../grasping")
sys.path.append("../motion")
import normals
from world_generator import save_world

import grasp
from grasp_database import *
from known_grippers import *

from loadObjs import *
from AntipodalGrasp import *
PHYSICS_SIMULATION = False  #not implemented correctly yet
if __name__ == '__main__':
    #load the robot / world file

    fn = "./main.xml"
    world = WorldModel()
    res = world.readFile(fn)
    robot = world.robot(0)
    qmin,qmax = robot.getJointLimits()
    for i in range(len(qmin)):
        if qmax[i] - qmin[i] > math.pi*2:
            qmin[i] = -float('inf')
            qmax[i] = float('inf')
    robot.setJointLimits(qmin,qmax)
    qstart = robot.getConfig()
    
    piece = loadPiece(world, "Pawn")[0]
    vis.add("world",world)

    print(piece)
    bmin,bmax = piece[1].geometry().getBBTight()
    print(bmin,bmax)
    gripper = robotiq_85_kinova_gen3
    db = GraspDatabase(gripper)
    try: db.load("../grasping/chess_grasps.json")
    except:
        print("Creating new grasp db")

    db.add_object(piece[1].getName())
    for i in range(world.numRigidObjects()):
        obj = world.rigidObject(i)
        #this will perform a reasonable center of mass / inertia estimate
        m = obj.getMass()
        m.estimate(obj.geometry(),mass=0.454,surfaceFraction=0.2)
        obj.setMass(m)

    t0 = time.time()
    grasps = antipodal_grasp_sample_volume(gripper,obj,10)
    t1 = time.time()
    print("Sampled grasps in",t1-t0,"s, min scoring grasp",grasps[0][1], "numgrasps:", len(grasps))
    for i,(g,s) in enumerate(grasps):
        db.add_grasp(piece[1], g.genGrasp(robot))
        name = "grasp{}".format(i)
        # u = math.exp(-(s-grasps[0][1])*2)
        # g.add_to_vis(name) 
    # print(db.object_to_grasps)
    db.save("chess_grasps.json")
    # print(qstart)
    # robot.setConfig(qstart)

    vis.add("pose", piece[1].getTransform())
    vis.run()

