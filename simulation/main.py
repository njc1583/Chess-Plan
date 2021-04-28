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
    robot = world.robot(0)
    #need to fix the spin joints somewhat
    qmin,qmax = robot.getJointLimits()
    for i in range(len(qmin)):
        if qmax[i] - qmin[i] > math.pi*2:
            qmin[i] = -float('inf')
            qmax[i] = float('inf')
    robot.setJointLimits(qmin,qmax)

    robot2 = world.robot(1)
    #need to fix the spin joints somewhat
    qmin,qmax = robot2.getJointLimits()
    for i in range(len(qmin)):
        if qmax[i] - qmin[i] > math.pi*2:
            qmin[i] = -float('inf')
            qmax[i] = float('inf')
    robot2.setJointLimits(qmin,qmax)

    chessEngine = ChessEngine(world, world.terrain('tabletop'))
    chessEngine.loadPieces()
    chessEngine.loadBoard()

    chessEngine.arrangeBoard(0)
    chessEngine.arrangePieces()

    chessEngine.visualizeBoardCorners(vis)

    # chessEngine.updateBoard()
    # chessEngine.updateBoard()
    # chessEngine.updateBoard()

    sensor = robot.sensor(0)

    xform = sensing.get_sensor_xform(sensor)
    link9_T = robot.link(9).getTransform()
    full_xform = se3.mul(link9_T, xform)

    # vis.add('Link 9 T', link9_T)
    # vis.add("Camera Xform", xform)
    # vis.add("Full xform", full_xform)

    table_center = chessEngine.getTableCenter()
    table_center = vectorops.add(table_center, [0,0,0.25])
    R_identity = so3.identity()

    print(xform)
    print(f'Sensor link: {sensor.getSetting("link")}')

    vis.add("world",world)

    motion = ChessMotion(world, robot, chessEngine)
    motion2 = ChessMotion(world, robot2, chessEngine)

    def main_loop_callback():
        if chessEngine.isTurnWhite():
            motion.loop_callback()
        else:
            motion2.loop_callback()
    vis.loop(callback=main_loop_callback)
