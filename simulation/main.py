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

    #load the gripper info and grasp database
    source_gripper = robotiq_85
    target_gripper = robotiq_85_kinova_gen3
    db = grasp_database.GraspDatabase(source_gripper)
    if not db.load("../data/grasps/robotiq_85_sampled_grasp_db.json"):
        raise RuntimeError("Can't load grasp database?")

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
    # vis.spin(100)

    # chessEngine.arrangeBoard(-90)
    # chessEngine.arrangePieces(randomlyRotatePieces=True)

    vis.add("world",world)
    # vis.spin(100)
    qstart = robot.getConfig()
    motion = ChessMotion(world, robot, chessEngine.boardTiles)
    # print(chessEngine.boardTiles)
    def planTriggered():
        robot.setConfig(qstart)
        path = motion.plan_to_square("E4")
        if path is not None:
            ptraj = trajectory.RobotTrajectory(robot,milestones=path)
            ptraj.times = [t / len(ptraj.times) * 5.0 for t in ptraj.times]
            #this function should be used for creating a C1 path to send to a robot controller
            traj = trajectory.path_to_trajectory(ptraj,timing='robot',smoothing=None)
            #show the path in the visualizer, repeating for 60 seconds
            vis.animate("start",traj)
            vis.add("traj",traj,endeffectors=[9])

    vis.addAction(planTriggered,"Plan to target",'p')
    # planTriggered()
    vis.run()
