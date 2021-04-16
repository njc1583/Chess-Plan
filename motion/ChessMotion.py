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
sys.path.append("../motion")
from known_grippers import robotiq_85_kinova_gen3
from motionHelpers import *
from planning import *
from motionGlobals import *

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

    def plan_to_square(self, square):#world,robot,object,gripper,grasp):
        tile = self.board[square]['tile']
        print("TILE:",tile.getTransform())
        return self.plan_pick_one(self.get_square_transform(square))
    def get_square_transform(self, square):
        tile = self.board[square]['tile']
        R,t = tile.getTransform()
        R_flip_y = so3.rotation([1,0,0],math.pi)
        R = so3.mul(R_flip_y,R) # makes gripper face downward
        t = vectorops.add(t, [0,0,0.2])# add 20 cm in z to avoid collision
        return (R,t)
    def go_to_square(self,square):
        return self.solve_robot_ik(self.get_square_transform(square))
    def check_collision(self):
        return is_collision_free_grasp(self.world, self.robot, self.currentObject)
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
    def plan_pick_one(self,Target):#world,robot,object,gripper,grasp):
        """
        Plans a picking motion for a given object and a specified grasp.

        Arguments:
            
            grasp (Grasp): the desired grasp. See common/grasp.py for more information.

        Returns:
            None or (transit,approach,lift): giving the components of the pick motion.
            Each element is a RobotTrajectory.  (Note: to convert a list of milestones
            to a RobotTrajectory, use RobotTrajectory(robot,milestones=milestones)

        Tip:
            vis.debug(q,world=world) will show a configuration.
        """
        qstart = self.robot.getConfig()

        # grasp.ik_constraint.robot = robot  #this makes it more convenient to use the ik module

        qgrasp = qstart
        solution = self.solve_robot_ik(Target)
        if not solution:
            self.robot.setConfig(qstart)
            print("Solution Failed")
            return None
        qgrasp = self.robot.getConfig()
        # qgrasp = grasp.set_finger_config(qgrasp)  #open the fingers the right amount
        qopen = self.gripper.set_finger_config(qgrasp,self.gripper.partway_open_config(1))   #open the fingers further
        distance = 0.2
        qpregrasp = retract(self.robot, self.gripper, vectorops.mul(self.gripper.primary_axis,-1*distance), local=True)   #TODO solve the retraction problem for qpregrasp?
        qstartopen = self.gripper.set_finger_config(qstart,self.gripper.partway_open_config(1))  #open the fingers of the start to match qpregrasp
        self.robot.setConfig(qstartopen)
        if qpregrasp is None:
            print("pregrasp failed")
            return None
        transit = feasible_plan(self.world,self.robot,qpregrasp)   #decide whether to use feasible_plan or optimizing_plan
        if not transit:
            print("transit failed")
            return None
        # Collision Checking
        for i in transit: #transit should be a list of configurations
            self.robot.setConfig(i)
            if not self.check_collision():
                return None

        self.robot.setConfig(qgrasp)
        qlift = retract(self.robot, self.gripper, vectorops.mul([0,0,1],distance), local=False) # move up a distance
        self.robot.setConfig(qstart)
        
        return (RobotTrajectory(self.robot,milestones=[qstart]+transit),RobotTrajectory(self.robot,milestones=[qpregrasp,qopen,qgrasp]),RobotTrajectory(self.robot,milestones=[qgrasp,qlift]))

