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

    qstart = robot.getConfig()
    motion = ChessMotion(world, robot, chessEngine.boardTiles)

    def planTriggered():
        global world, robot, sensor
        robot.setConfig(qstart)

        qconfig = motion.point_camera_at_board(world, robot, 9, chessEngine)

        if qconfig is None:
            print("Unable to find an angle approach at any angle")
        else:
            robot.setConfig(qconfig)
            vis.update()


        xformnobot = sensing.get_sensor_xform(sensor)

        print(f'xform w/o robot: {xformnobot}')

        vis.add('xformnobot', xformnobot)

        xformbot = sensing.get_sensor_xform(sensor, robot)

        print(f'xform w/ robot: {xformbot}')

        vis.add('xformbot', xformbot)

        # link9_T = robot.link(9).getTransform()

        # Tgripper = (so3.identity(), [0,0,0.1])

        # full_xform = se3.mul(link9_T, Tgripper)

        # sensing.set_sensor_xform(sensor, full_xform)

        # vis.add('xform', xform)

        # vis.add('full_xform', full_xform)

        # sensing.set_sensor_xform(sensor, full_xform)

        sensor.kinematicReset()
        sensor.kinematicSimulate(world, 0.01)

        chessEngine.updateBoard()
        # a = input("Help me: ")
        
        rgb, depth = sensing.camera_to_images(sensor)        
        Image.fromarray(rgb).save('test.jpg')

        # global world,robot
        # robot.setConfig(qstart)
        # square = "h8"
        # path = motion.plan_to_square(square)
        # if path is None:
        #     print("Unable to plan pick")
        # else:
        #     (transit,approach,lift) = path
        #     traj = transit
        #     traj = traj.concat(approach,relative=True,jumpPolicy='jump')
        #     traj = traj.concat(lift,relative=True,jumpPolicy='jump')
        #     vis.add("traj",traj,endEffectors=[9])
        #     vis.animate(vis.getItemName(robot),traj)
        # tTarget = motion.get_square_transform(square)
        # vis.add("targetTransform", tTarget)
    
    
    vis.addAction(planTriggered,"Plan to target",'p')

    vis.run()
