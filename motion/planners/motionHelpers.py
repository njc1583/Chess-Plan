from klampt.model import ik
import sys
sys.path.append('../common')
import gripper
import known_grippers
from klampt.math import vectorops,so3,se3

sys.path.append("../motion/planners")
from motionGlobals import *
finger_pad_links = ['gripper:Link_4','gripper:Link_6']

def is_collision_free_grasp(world,robot,obj, place=False):
    if robot.selfCollides():
        return False
    for i in range(world.numTerrains()):
        for j in range(robot.numLinks()):
            if robot.link(j).geometry().collides(world.terrain(i).geometry()):
                if DEBUG_MODE:
                    print("Robot-Terrain Collision Detected:", robot.link(j).getName(), j, "With", world.terrain(i).getName())
                return False
    for i in range(world.numRigidObjects()):
        for j in range(robot.numLinks()):
            if obj and i == obj.index:
                if robot.link(j).getName() not in finger_pad_links and robot.link(j).geometry().collides(obj.geometry()):
                    if DEBUG_MODE:
                        print("Grasped Object Collision Detected:", robot.link(j).getName(), j, "With", obj.getName())
                    return False
            elif robot.link(j).geometry().collides(world.rigidObject(i).geometry()):
                if DEBUG_MODE:
                    print("RigidObject-Robot Collision Detected:", robot.link(j).getName(), j, "With", world.rigidObject(i).getName())
                return False
    if place:
        for i in range(world.numTerrains()):
            if obj and obj.geometry().collides(world.terrain(i).geometry()):
                if DEBUG_MODE:
                    print("Grasped Object-Terrain Collision Detected:", world.terrain(i).getName(), j, "With", obj.getName())
                return False
        for i in range(world.numRigidObjects()):
            if obj and i != obj.index and obj.geometry().collides(world.rigidObject(i).geometry()):
                if DEBUG_MODE:
                    print("Grasped Object-Object Collision Detected:", world.rigidObject(i).getName(), j, "With", obj.getName())
                return False
    return True

def retract(robot,gripper,amount,local=True):
    """Retracts the robot's gripper by a vector `amount`.

    if local=True, amount is given in local coordinates.  Otherwise, its given in
    world coordinates.
    """
    if not isinstance(gripper,(int,str)):
        gripper = gripper.base_link
    link = robot.link(gripper)
    Tcur = link.getTransform()
    if local:
        amount = so3.apply(Tcur[0],amount)
    obj = ik.objective(link,R=Tcur[0],t=vectorops.add(Tcur[1],amount))
    solution = ik.solve(obj)
    if solution:
        return robot.getConfig()
    else:
        print("Retract IK failed")
        print("Final config:",robot.getConfig())
        global DEBUG_MODE
        if DEBUG_MODE:
            return robot.getConfig()
        else:
            return None