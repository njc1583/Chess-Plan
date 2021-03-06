from klampt import WorldModel
from klampt import vis 
import math
import time
import sys
sys.path.append("../common")

import grasp
from grasp_database import *
from known_grippers import *
sys.path.append("../grasping")
from loadObjs import *
from AntipodalGrasp import *
import os
if os.path.exists("chess_grasps.json"):
  os.remove("chess_grasps.json")
PHYSICS_SIMULATION = False  #not implemented correctly yet
VIEW_GRASPS = False # Visualizes grasps when True, writes to grasp db otherwise
if __name__ == '__main__':
    # Load the robot / world file
    fn = "./main.xml"
    gripper = robotiq_85_kinova_gen3
    db = GraspDatabase(gripper)
    try: db.load("../grasping/chess_grasps.json")
    except:
        print("Creating new grasp db")
    piece_names = ["Pawn","Rook","Bishop","Knight","Queen","King"]
    for name in piece_names:
        world = WorldModel()
        if not VIEW_GRASPS:
            res = world.readFile(fn)
            if not res:
                print("Could not open world file")
                exit(0)
            # Initialize Robot
            robot = world.robot(0)
            qmin,qmax = robot.getJointLimits()
            for i in range(len(qmin)):
                if qmax[i] - qmin[i] > math.pi*2:
                    qmin[i] = -float('inf')
                    qmax[i] = float('inf')
            robot.setJointLimits(qmin,qmax)
        vis.add("world",world)

        piece = loadPiece(world, name)[0][1]
        for i in range(world.numRigidObjects()):
            obj = world.rigidObject(i)
            #this will perform a reasonable center of mass / inertia estimate
            m = obj.getMass()
            m.estimate(obj.geometry(),mass=0.454,surfaceFraction=0.2)
            obj.setMass(m)


        if piece.getName() not in db.object_to_grasps.keys():
            db.add_object(piece.getName())

        t0 = time.time()
        grasps = antipodal_grasp_sample_volume(gripper,piece,50)
        t1 = time.time()
        print("Sampled grasps in",t1-t0,"s, min scoring grasp",grasps[0][1], "numgrasps:", len(grasps))
        maxScore = 0
        for i,(g,s) in enumerate(grasps):
            if not VIEW_GRASPS:
                db.add_grasp(piece, g.genGrasp(robot))
            maxScore = max(maxScore,g.finger_width)
            if VIEW_GRASPS:
                name = "grasp{}".format(i)
                u = math.exp(-(s-grasps[0][1])*2)
                g.add_to_vis(name) 
        print("Piece:",piece.getName(), "maxScore:", maxScore)
        if VIEW_GRASPS:
            vis.add(piece.getName(),piece)
            vis.run()
    db.save("chess_grasps.json")


