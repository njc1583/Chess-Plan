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
from klampt.model import sensing

sys.path.append("../common")
sys.path.append("../engines")
sys.path.append("../motion")

from PIL import Image

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
    robot_white = world.robot(0)

    #need to fix the spin joints somewhat
    qmin,qmax = robot_white.getJointLimits()
    for i in range(len(qmin)):
        if qmax[i] - qmin[i] > math.pi*2:
            qmin[i] = -float('inf')
            qmax[i] = float('inf')
    robot_white.setJointLimits(qmin,qmax)

    robot_black = world.robot(1)
    #need to fix the spin joints somewhat
    qmin,qmax = robot_black.getJointLimits()
    for i in range(len(qmin)):
        if qmax[i] - qmin[i] > math.pi*2:
            qmin[i] = -float('inf')
            qmax[i] = float('inf')
    robot_black.setJointLimits(qmin,qmax)

    chessEngine = ChessEngine(world, world.terrain('tabletop'))
    chessEngine.loadPieces()
    chessEngine.loadBoard()

    chessEngine.arrangeBoard(-90)
    chessEngine.arrangePieces()

    for i in range(world.numRigidObjects()):
        world.rigidObject(i).appearance().setSilhouette(0)

    # chessEngine.visualizeBoardCorners(True, vis)

    # chessEngine.visualizeTiles(vis)

    # table_center = chessEngine.getTableCenter()
    # vis.add('Table Center', table_center)

    vis.add("world",world)

    motion_white = ChessMotion(world, robot_white, True, chessEngine, isAuto=True)
    motion_black = ChessMotion(world, robot_black, False, chessEngine, isAuto=True)

    # motion_white.visualize_rotation_points(table_center, 45, 90, vis)
    # motion_black.visualize_rotation_points(table_center, 45, 90, vis)

    def main_loop_callback():
        if chessEngine.startGame:
            if chessEngine.isTurnWhite():
                motion_white.loop_callback()
            else:
                motion_black.loop_callback()

    def start():
        chessEngine.startGame = True

    vis.addAction(start,"Start Chess-Plan",'s')

    vis.loop(callback=main_loop_callback)
