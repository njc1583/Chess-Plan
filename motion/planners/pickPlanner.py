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
from multiStepPlanner import *
from motionGlobals import *
from motionHelpers import *
from planning import *
class PickPlanner(MultiStepPlanner):
    """For problem 2C
    """
    def __init__(self,world,robot,object,gripper,grasps):
        MultiStepPlanner.__init__(self,['grasp','qgrasp','approach','transit','lift'])
        self.qstart = robot.getConfig()
        self.world=world
        self.robot=robot
        self.object=object
        self.gripper=gripper
        self.grasps=grasps
    def check_collision(self):
        return is_collision_free_grasp(self.world, self.robot, self.object)
    def solve_qgrasp(self,grasp):
        #TODO: solve for the grasping configuration
        qstart = self.robot.getConfig()
        solution = ik.solve_global(grasp.ik_constraint, iters=100, numRestarts = 10, feasibilityCheck=self.check_collision)
        if not solution:
            self.robot.setConfig(qstart)
            return None
        print(f"Solution: {solution}")
        qgrasp = self.robot.getConfig()
        qgrasp = grasp.set_finger_config(qgrasp)
        return qgrasp

    def solve_approach(self,grasp,qgrasp):
        #TODO: solve for the approach
        distance = 0.1
        qpregrasp = retract(self.robot, self.gripper, vectorops.mul(self.gripper.primary_axis,-1*distance), local=True)
        qopen = self.gripper.set_finger_config(qgrasp,self.gripper.partway_open_config(grasp.score + 0.1))   #open the fingers further

        return [qpregrasp,qopen,qgrasp]

    def solve_transit(self,qpregrasp):
        #TODO: solve for the transit path
        if qpregrasp == None:
            return None
        qstartopen = self.gripper.set_finger_config(self.qstart,self.gripper.partway_open_config(1))  #open the fingers of the start to match qpregrasp
        self.robot.setConfig(qstartopen)
        transit = feasible_plan(self.world,self.robot,qpregrasp)   #decide whether to use feasible_plan or optimizing_plan
        if not transit:
            return None

        for i in transit: #transit should be a list of configurations
            self.robot.setConfig(i)
            if not self.check_collision():
                return None
        return transit

    def solve_lift(self,qgrasp):
        #TODO: solve for the lifting configurations
        self.robot.setConfig(qgrasp)
        distance = 0.1
        qlift = retract(self.robot, self.gripper, vectorops.mul([0,0,1],distance), local=False) # move up a distance
        self.robot.setConfig(self.qstart)
        return qlift

    def solve_item(self,plan,item):
        """Returns a pair (status,children) where status is one of the codes FAIL,
        COMPLETE, CHILDREN, CHILDREN_AND_SELF, and children is a list of solutions
        that complete more of the plan.
        """
        if item == 'grasp':
            print("Assigning grasps")
            return StepResult.CHILDREN,self.grasps
        if item == 'qgrasp':
            print("Planning IK configuration")
            grasp = plan['grasp']
            result = self.solve_qgrasp(grasp)
            if result is None:
                print("IK solve failed... trying again")
                return StepResult.CONTINUE,[]
            else:
                print("IK solve succeeded, moving on to pregrasp planning")
                return StepResult.CHILDREN_AND_CONTINUE,[result]
        if item == 'approach':
            print("Planning approach")
            grasp = plan['grasp']
            qgrasp = plan['qgrasp']
            result = self.solve_approach(grasp,qgrasp)
            if result is None:
                return StepResult.FAIL,[]
            return StepResult.CHILDREN,[result]
        if item == 'transit':
            print("Transit planning")
            qpregrasp = plan['approach'][0]
            result = self.solve_transit(qpregrasp)
            if result is None:
                print("Transit planning failed")
                return StepResult.CONTINUE,[]
            else:
                print("Transit planning succeeded!")
                return StepResult.CHILDREN,[result]
        if item == 'lift':
            qgrasp = plan['qgrasp']
            result = self.solve_lift(qgrasp)
            if result is None:
                return StepResult.FAIL,[]
            return StepResult.CHILDREN,[result]
        raise ValueError("Invalid item "+item)


    def score(self,plan):
        """Priority score for a partial plan"""
        #TODO: prioritize grasps with low score
        
        return plan['grasp'].score + MultiStepPlanner.score(self,plan)

    def assemble_result(self,plan):
        """Get the results from a partial plan"""
        qstart = self.qstart
        transit = plan['transit']
        approach = plan['approach']        
        qgrasp = plan['qgrasp']
        qlift = plan['lift']
        #TODO: construct the RobotTrajectory triple as in plan_pick_one
        return (RobotTrajectory(self.robot,milestones=[qstart]+transit),RobotTrajectory(self.robot,milestones=approach),RobotTrajectory(self.robot,milestones=[qgrasp,qlift]))

    
def plan_pick_multistep(world,robot,object,gripper,grasps):
    """
    Plans a picking motion for a given object and a set of possible grasps, sorted
    in increasing score order.

    Arguments:
        world (WorldModel): the world, containing robot, object, and other items that
            will need to be avoided.
        robot (RobotModel): the robot in its current configuration
        object (RigidObjectModel): the object to pick.
        gripper (GripperInfo): the gripper.
        grasp (Grasp): the desired grasp. See common/grasp.py for more information.

    Returns:
        None or (transit,approach,lift): giving the components of the pick motion.
        Each element is a RobotTrajectory.  (Note: to convert a list of milestones
        to a RobotTrajectory, use RobotTrajectory(robot,milestones=milestones)

    Tip:
        vis.debug(q,world=world) will show a configuration.
    """
    qstart = robot.getConfig()
    for grasp in grasps:
        grasp.ik_constraint.robot = robot  #this makes it more convenient to use the ik module
    planner = PickPlanner(world,robot,object,gripper,grasps)
    time_limit = 60
    return planner.solve(time_limit)