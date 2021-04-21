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
    fn = "./main.xml"
    world = WorldModel()
    res = world.readFile(fn)
    print(res)
    if not res:
        print("Unable to read file",fn)
        exit(0)
    for i in range(world.numRigidObjects()):
        obj = world.rigidObject(i)
        print(obj.getName(),obj.index)
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

    chessEngine.visualizeBoardCorners(vis)

    vis.add("world",world)

    qstart = robot.getConfig()
    motion = ChessMotion(world, robot, chessEngine.boardTiles)
    solved_trajectory = None
    trajectory_is_transfer = None
    def planTriggered():
        global world, robot, motion, solved_trajectory, trajectory_is_transfer
        solved_trajectory,trajectory_is_transfer = motion.construct_trajectory("e4")

    vis.addAction(planTriggered,"Plan to target",'p')

    executing_plan = False
    execute_start_time = None
    def executePlan():
        global solved_trajectory,trajectory_is_transfer,executing_plan,execute_start_time
        if solved_trajectory is None:
            return
        executing_plan = True
        execute_start_time = time.time()

    vis.addAction(executePlan,"Execute plan",'e')

    def loop_callback():
        global motion, solved_trajectory, trajectory_is_transfer
        if not motion.executing_plan:
            san = input("Enter Chess Move:")
            solved_trajectory,trajectory_is_transfer = motion.make_move(san)
            return
        t = time.time()-motion.execute_start_time
        vis.addText("time","Time %.3f"%(t),position=(10,10))
        qcurrent = solved_trajectory.eval(t)
        robot.setConfig(qcurrent)
        during_transfer = trajectory_is_transfer.eval(t)[0]
        if during_transfer:
            motion.currentObject.setTransform(*se3.mul(robot.link(9).getTransform(),motion.Tobject_gripper))
        if t > solved_trajectory.duration():
            motion.executing_plan = False
            solved_trajectory = None
            robot.setConfig(qstart)

    vis.loop(callback=motion.loop_callback)
