from klampt.plan import robotplanning
from klampt.plan.cspace import MotionPlan
from klampt.model import trajectory
from klampt.model.trajectory import Trajectory,RobotTrajectory, execute_path,path_to_trajectory
from klampt import vis 
from klampt.model import ik
from klampt.math import vectorops,so3,se3
from klampt import RobotModel
import math
import time
import sys
sys.path.append('../common')
from known_grippers import robotiq_85_kinova_gen3
sys.path.append("../motion/planners")
from pickPlanner import *
from motionGlobals import *
import grasp_database

class ChessMotion:
    """
        Executes pick-and-place motion plans on the chessboard
        Attributes:
            world (WorldModel): the world, containing robot, object, and other items that
                will need to be avoided.
            robot (RobotModel): the robot in its current configuration
            qstart(RobotConfig list): initial robot start config
            gripper (GripperInfo): the gripper
            board (dict): boardTiles object from ChessEngine
            currentObject (RigidObjectModel): the chess piece object to pick up currently
    """
    def __init__(self, world, robot, board):
        self.world = world
        self.robot = robot
        self.qstart = robot.getConfig()
        self.gripper = robotiq_85_kinova_gen3
        self.board = board
        self.currentObject = None
        self.db = grasp_database.GraspDatabase(self.gripper)
        if not self.db.load("../grasping/chess_grasps.json"):
            raise RuntimeError("Can't load grasp database?")
    def plan_to_square(self, square):#world,robot,object,gripper,grasp):
        tile = self.board[square]['tile']
        print("TILE:",tile.getTransform())
        return self.plan_pick_grasps(self.get_square_transform(square))
    def get_square_transform(self, square):
        tile = self.board[square]['tile']
        R,t = tile.getTransform()
        Tobj = obj.getTransform()
        Tobject_grasp = se3.mul(se3.inv(gripper_link.getTransform()),Tobj)
        R_flip_y = so3.rotation([1,0,0],math.pi)
        R = so3.mul(R_flip_y,R) # makes gripper face downward
        t = vectorops.add(t, [0,0,0.2])# add 20 cm in z to avoid collision
        return (R,t)
    def get_object_grasps(self, name, T_obj):
        orig_grasps = self.db.object_to_grasps[name]
        grasps = [grasp.get_transformed(T_obj) for grasp in orig_grasps]
        return grasps
    def plan_to_piece(self,square):
        piece = self.board[square]['piece'][1]
        self.currentObject = piece
        print(piece)
        name = piece.getName().split('_')[0]
        grasps = self.get_object_grasps(name, piece.getTransform())
        path = plan_pick_multistep(self.world,self.robot,self.currentObject,self.gripper,grasps)
        return path
    def plan_to_place(self,square):
        
    def go_to_square(self,square):
        return self.solve_robot_ik(self.get_square_transform(square))

    def solve_robot_ik(self,Tgripper):
        """Given a robot, a gripper, and a desired gripper transform,
        solve the IK problem to place the gripper at the desired transform.
        Returns:
            list or None: Returns None if no solution was found, and
            returns an IK-solving configuration q otherwise.
        Args:
            Tgripper (klampt se3 object)
        """
        link = self.robot.link(self.gripper.base_link)
        goal = ik.objective(link,R=Tgripper[0],t=Tgripper[1])
        # solver = ik.solver(goal)
        solution = ik.solve_global(goal, iters=100, numRestarts = 10, feasibilityCheck=self.check_collision)
        if solution:
            print("Hooray, IK solved")
            print("Resulting config:",self.robot.getConfig())
            return self.robot.getConfig()
        else:
            print("IK failed")
            print("Final config:",self.robot.getConfig())
            global DEBUG_MODE
            if DEBUG_MODE:
                return self.robot.getConfig()
            else:
                return None
