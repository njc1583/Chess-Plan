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
sys.path.append("../engines")
sys.path.append("../motion")

from world_generator import save_world

import grasp
import grasp_database
from known_grippers import *

from ChessEngine import ChessEngine
from ChessMotion import ChessMotion

PHYSICS_SIMULATION = False  #not implemented correctly yet
if __name__ == '__main__':
    #load the robot / world file
    # fn = "./main.xml"
    fn = "./worlds/default.xml"
    world = WorldModel()
    res = world.readFile(fn)
    if not res:
        fn = "./main.xml"
        world = WorldModel()
        res = world.readFile(fn)
        chessEngine = ChessEngine(world, world.terrain('tabletop'))
        chessEngine.loadPieces()
        chessEngine.loadBoard()
        chessEngine.arrangeBoard()
        chessEngine.arrangePieces()
        print(chessEngine.pieces)
    if not res:
        print("Unable to read file",fn)
        exit(0)
    for i in range(world.numRigidObjects()):
        obj = world.rigidObject(i)
        #this will perform a reasonable center of mass / inertia estimate
        m = obj.getMass()
        m.estimate(obj.geometry(),mass=0.454,surfaceFraction=0.2)
        obj.setMass(m)
    
    robot = world.robot(0)
    #need to fix the spin joints somewhat
    qmin,qmax = robot.getJointLimits()
    for i in range(len(qmin)):
        if qmax[i] - qmin[i] > math.pi*2:
            qmin[i] = -float('inf')
            qmax[i] = float('inf')
    robot.setJointLimits(qmin,qmax)

    chessEngine = ChessEngine(world, world.terrain('tabletop'))
    chessEngine.loadPieces()
    chessEngine.loadBoard()

    chessEngine.arrangeBoard(0)
    chessEngine.arrangePieces()

    vis.add("world",world)

    qstart = robot.getConfig()
    motion = ChessMotion(world, robot, chessEngine.boardTiles)

    def planTriggered():
        global world,robot
        robot.setConfig(qstart)
        square = "G8"
        path = motion.plan_to_piece(square)
        if path is None:
            print("Unable to plan pick")
        else:
            (transit,approach,lift) = path
            traj = transit
            traj = traj.concat(approach,relative=True,jumpPolicy='jump')
            traj = traj.concat(lift,relative=True,jumpPolicy='jump')
            vis.add("traj",traj,endEffectors=[9])
            vis.animate(vis.getItemName(robot),traj)
        # tTarget = motion.get_square_transform(square)
        # vis.add("targetTransform", tTarget)
    vis.addAction(planTriggered,"Plan to target",'p')

    vis.run()
