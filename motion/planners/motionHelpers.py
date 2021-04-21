from klampt.model import ik
import sys
sys.path.append('../common')
import gripper
import known_grippers
from klampt.math import vectorops,so3,se3

sys.path.append("../motion/planners")
from motionGlobals import *
finger_pad_links = ['gripper:Link 4','gripper:Link 6']

def is_collision_free_grasp(world,robot,object):
    if robot.selfCollides():
        return False
    for i in range(world.numTerrains()):
        for j in range(robot.numLinks()):
            if robot.link(j).geometry().collides(world.terrain(i).geometry()):
                return False
    for i in range(world.numRigidObjects()):
        for j in range(robot.numLinks()):
            if robot.link(j).geometry().collides(world.rigidObject(i).geometry()):
                return False
    if object:
        for j in range(robot.numLinks()):
            if robot.link(j).getName() not in finger_pad_links and robot.link(j).geometry().collides(object.geometry()):
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