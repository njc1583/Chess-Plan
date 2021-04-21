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
        robot.setConfig(qstart)
        square = 'D1'#input("Square:")
        path = motion.plan_to_piece(square)
        if path is None:
            print("Unable to plan pick")
        else:
            trajectory_is_transfer = Trajectory()
            trajectory_is_transfer.times.append(0)
            trajectory_is_transfer.milestones.append([0])

            (transit,approach) = path
            traj = transit
            traj = traj.concat(approach,relative=True,jumpPolicy='jump')
            trajectory_is_transfer.times.append(traj.endTime())
            trajectory_is_transfer.times.append(traj.endTime())
            trajectory_is_transfer.milestones.append([0])
            trajectory_is_transfer.milestones.append([1])
            robot.setConfig(approach.milestones[-1])
            target_square = 'A4' #input("Target Square:")
            tTarget = motion.get_target_transform(target_square)
            vis.add("targetTransform", tTarget)
            print("attempting plan to place")
            res = motion.plan_to_place(target_square)
            if res is None:
                print("Unable to plan place")
            else:
                (transfer,lower,retract) = res
                traj = traj.concat(transfer,relative=True,jumpPolicy='jump')
                traj = traj.concat(lower,relative=True,jumpPolicy='jump')
                trajectory_is_transfer.times.append(traj.endTime())
                trajectory_is_transfer.times.append(traj.endTime())
                trajectory_is_transfer.milestones.append([1])
                trajectory_is_transfer.milestones.append([0])
                traj = traj.concat(retract,relative=True,jumpPolicy='jump')
                trajectory_is_transfer.times.append(traj.endTime())
                trajectory_is_transfer.milestones.append([0])
                solved_trajectory = traj
            vis.add("traj",traj,endEffectors=[9])
            robot.setConfig(qstart)

            # vis.animate(vis.getItemName(robot),traj)
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

    was_grasping = False
    print("Num objs orig:",world.numRigidObjects())
    def loop_callback():
        global was_grasping,solved_trajectory,trajectory_is_transfer,executing_plan,execute_start_time,qstart
        if not executing_plan:
            return
        t = time.time()-execute_start_time
        vis.addText("time","Time %.3f"%(t),position=(10,10))
        qstart = solved_trajectory.eval(t)
        robot.setConfig(qstart)
        during_transfer = trajectory_is_transfer.eval(t)[0]
        if during_transfer:
            print("Num objs:",world.numRigidObjects())
            motion.currentObject.setTransform(*se3.mul(robot.link(9).getTransform(),motion.Tobject_gripper))
        if t > solved_trajectory.duration():
            executing_plan = False
            solved_trajectory = None

    vis.loop(callback=loop_callback)
