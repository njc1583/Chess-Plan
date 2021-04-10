from klampt.plan import robotplanning
from klampt.plan.cspace import MotionPlan
from klampt.model import trajectory
from klampt import vis 
from klampt import RobotModel
import math
import time
import sys
sys.path.append('../common')
import gripper
import known_grippers
from klampt.math import vectorops,so3,se3

class ChessMotion:
    def __init__(self, world, robot, board):
        self.world = world
        self.robot = robot
        self.gripper = robot.link(9)
        self.board = board      # boardTiles object from ChessEngine
        self.currentObject = None

    def plan_to_square(self, square):#world,robot,object,gripper,grasp):
        pass
        
    
