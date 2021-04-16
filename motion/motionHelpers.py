from klampt.model import ik
import sys
sys.path.append('../common')
import gripper
import known_grippers
from klampt.math import vectorops,so3,se3

sys.path.append("../motion")
from motionGlobals import *
finger_pad_links = ['gripper:Link 4','gripper:Link 6']
gripper_center = vectorops.madd(known_grippers.robotiq_85.center,known_grippers.robotiq_85.primary_axis,known_grippers.robotiq_85.finger_length-0.005)
gripper_closure_axis = known_grippers.robotiq_85.secondary_axis

class simpleGrasp:
    def __init__(self, center, axis = [1,0,0], width=0.5) -> None:
        self.center = center # Piece center for a grasp
        self.axis = axis # should be able to use constant horizontal axis for any piece
        self.width = width # width to close fingers to, maybe use ray cast stuff from MP3??

def match_grasp(gripper_center,gripper_closure_axis,grasp):
    """
    Args:
        gripper_center (3-vector): local coordinates of the center-point between the gripper's fingers.
        gripper_closure_axis (3-vector): local coordinates of the axis connecting the gripper's fingers.
        grasp (AntipodalGrasp): the desired grasp
        
    Returns:
        (R,t): a Klampt se3 element describing the maching gripper transform
    """
    R_grip = so3.canonical(gripper_closure_axis) # world to gripper axis
    R_grasp = so3.canonical(grasp.axis) # world to grasp axis
    t = vectorops.sub(grasp.center,gripper_center)
    R_final = so3.mul(R_grasp,so3.inv(R_grip)) # gripper to grasp
    #print(se3.apply((R_final,t), gripper_center), grasp.center)
    return (R_final,t)

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