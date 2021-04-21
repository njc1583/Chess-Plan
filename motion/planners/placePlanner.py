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
import random

APPROACH_DIST = 0.2
class PlacePlanner(MultiStepPlanner):
    """
    Plans a placing motion for a given object and a specified grasp.

    Arguments:
        world (WorldModel): the world, containing robot, object, and other items that
            will need to be avoided.
        robot (RobotModel): the robot in its current configuration
        object (RigidObjectModel): the object to pick.
        Tobject_gripper (se3 object): transform of the object with respect to the gripper..
        goal_bounds (list): bounds of the goal region [(xmin,ymin,zmin,(xmax,ymax,zmax)]

    Returns:
        None or (transfer,lower,ungrasp): giving the components of the place motion.
        Each element is a RobotTrajectory.  (Note: to convert a list of milestones
        to a RobotTrajectory, use RobotTrajectory(robot,milestones=milestones)

    Tip:
        vis.debug(q,world=world) will show a configuration.
    """
    def __init__(self,world,robot,object,Tobject_gripper,gripper,target=None,goal_bounds=None):
        MultiStepPlanner.__init__(self,['lift','placement','qplace','qpreplace','retract','transfer'])
        self.world=world
        self.robot=robot
        self.object=object
        self.Tobject_gripper=Tobject_gripper
        self.gripper=gripper
        self.goal_bounds=goal_bounds
        self.T_target = target
        object.setTransform(*se3.identity())
        self.objbb = object.geometry().getBBTight()
        self.qstart = robot.getConfig()  #grasped object 

    def object_free(self,q):
        """Helper: returns true if the object is collision free at configuration q, if it is
        attached to the gripper."""
        self.robot.setConfig(q)
        gripper_link = self.robot.link(self.gripper.base_link)
        self.object.setTransform(*se3.mul(gripper_link.getTransform(),self.Tobject_gripper))
        for i in range(self.world.numTerrains()):
            if self.object.geometry().collides(self.world.terrain(i).geometry()):
                return False
        for i in range(self.world.numRigidObjects()):
            if i == self.object.index: continue
            if self.object.geometry().collides(self.world.rigidObject(i).geometry()):
                return False
        return True

    def solve_placement(self):
        """Implemented for you: come up with a collision-free target placement"""
        if self.T_target is not None:
            print("Used given target transform")
            print(self.T_target)
            return [self.T_target]
        if self.goal_bounds is None:
            print("No Goal Bounds")
            return None
        obmin,obmax = self.objbb
        ozmin = obmin[2]
        min_dims = min(obmax[0]-obmin[0],obmax[1]-obmin[1])
        center = [0.5*(obmax[0]+obmin[0]),0.5*(obmax[1]-obmin[1])]
        xmin,ymin,zmin = self.goal_bounds[0]
        xmax,ymax,zmax = self.goal_bounds[1]
        center_sampling_range = [(xmin+min_dims/2,xmax-min_dims/2),(ymin+min_dims/2,ymax-min_dims/2)]
        Tobj_feasible = []
        for iters in range(20):
            crand = [random.uniform(b[0],b[1]) for b in center_sampling_range]
            Robj = so3.rotation((0,0,1),random.uniform(0,math.pi*2))
            tobj = vectorops.add(so3.apply(Robj,[-center[0],-center[1],0]),[crand[0],crand[1],zmin-ozmin+0.002])
            self.object.setTransform(Robj,tobj)
            feasible = True
            for i in range(self.world.numTerrains()):
                if self.object.geometry().collides(self.world.terrain(i).geometry()):
                    feasible=False
                    break
            if not feasible:
                bmin,bmax = self.object.geometry().getBBTight()
                if bmin[0] < xmin:
                    tobj[0] += xmin-bmin[0]
                if bmax[0] > xmax:
                    tobj[0] -= bmin[0]-xmax
                if bmin[1] < ymin:
                    tobj[1] += ymin-bmin[1]
                if bmax[1] > ymax:
                    tobj[1] -= bmin[1]-ymax
                self.object.setTransform(Robj,tobj)
                feasible = True
                for i in range(self.world.numTerrains()):
                    if self.object.geometry().collides(self.world.terrain(i).geometry()):
                        feasible=False
                        break
                if not feasible:
                    continue
            for i in range(self.world.numRigidObjects()):
                if i == self.object.index: continue
                if self.object.geometry().collides(self.world.rigidObject(i).geometry()):
                    #raise it up a bit
                    bmin,bmax = self.world.rigidObject(i).geometry().getBB()
                    tobj[2] = bmax[2]-ozmin+0.002
                    self.object.setTransform(Robj,tobj)
            Tobj_feasible.append((Robj,tobj))
        print("Found",len(Tobj_feasible),"valid object placements")
        return Tobj_feasible
    def check_collision(self,newConfig=None):
        qstart = self.robot.getConfig()
        if newConfig is not None:
            self.robot.setConfig(newConfig)        # Set new robot config
        obj_start = self.object.getTransform()
        gripper_link = self.robot.link(9)
        T_grip_w = gripper_link.getTransform()
        T_obj_w = se3.mul(T_grip_w,self.Tobject_gripper)
        self.object.setTransform(*T_obj_w)        # Set new object config based on gripper pose
        ret = is_collision_free_grasp(self.world, self.robot, self.object) # Check new arm position is valid
        for i in range(self.world.numTerrains()): # Check object to cabinet/robot stand collisions
            if self.object.geometry().collides(self.world.terrain(i).geometry()):
                print("obj collides with terrain!")
                ret = False
        for i in range(self.world.numRigidObjects()):
            if i == self.object.index: continue
            if self.object.geometry().collides(self.world.rigidObject(i).geometry()):
                print("Cur obj",self.object.getName(), self.object.index, \
                    "collides with object:", self.world.rigidObject(i).getName(), i)
                ret = False
        #vis.debug(self.robot)
        # Reset robot/object transform
        self.object.setTransform(*obj_start)
        self.robot.setConfig(qstart)
        return ret
    def solve_lift(self):
        #TODO: solve for the lifting configurations
        self.robot.setConfig(self.qstart)
        distance = APPROACH_DIST
        qlift = retract(self.robot, self.gripper, vectorops.mul([0,0,1],distance), local=False) # move up a distance
        self.robot.setConfig(self.qstart)
        return qlift
    def solve_qplace(self,Tplacement):
        #TODO: adjust ik objective to allow rotation about z for placement configuration
        if not isinstance(self.gripper,(int,str)):
            temp_gripper = self.gripper.base_link
        link = self.robot.link(temp_gripper)
        T_grip = se3.mul(Tplacement,se3.inv(self.Tobject_gripper))
        obj = ik.objective(link,R=T_grip[0],t=T_grip[1])
        #res = ik.solve(obj)
        res = ik.solve_global(obj, iters=100, numRestarts = 10, feasibilityCheck=self.check_collision)
        if not res:
            global DEBUG_MODE
            if DEBUG_MODE:
                return self.robot.getConfig()
            else:
                return None
        return self.robot.getConfig()

    def solve_preplace(self,qplace):
        #TODO: solve for the preplacement configuration
        self.robot.setConfig(qplace)
        distance = APPROACH_DIST
        qpreplace = retract(self.robot, self.gripper, vectorops.mul(self.gripper.primary_axis,-1*distance), local=True)
        self.robot.setConfig(self.qstart)
        return qpreplace

    def solve_retract(self,qplace):
        #TODO: solve for the retraction step
        self.robot.setConfig(qplace)
        amount = self.gripper.config_to_opening(self.gripper.get_finger_config(qplace))
        qopen = self.gripper.set_finger_config(qplace,self.gripper.partway_open_config(amount + 0.1))   #open the fingers further
        distance = 0.2
        self.robot.setConfig(qopen)
        qlift = retract(self.robot, self.gripper, vectorops.mul([0,0,1],distance), local=False) # move up a distance
        self.robot.setConfig(self.qstart)
        return [qopen,qlift]

    def solve_transfer(self,qpreplace,qlift):
        #TODO: solve for the transfer plan
        moving_joints = [1,2,3,4,5,6,7]
        gripper_link = 9
        self.robot.setConfig(qlift)
        if qpreplace is None:
            return None

        planOpts = {'type':'sbl','perturbationRadius':0.5,'shortcut':True,'restart':True,'restartTermCond':"{foundSolution:1,maxIters:100}"}
        plan = robotplanning.planToConfig(self.world, self.robot, qpreplace, edgeCheckResolution=1e-2, 
                                            extraConstraints=[self.check_collision],
                                            movingSubset=moving_joints, **planOpts)
        if plan == None:
            print("Planning Failed")
            return None
        numIters = 80
        t1 = time.time()
        t0 = time.time()
        path = []
        while(t1-t0 < 60 and (path == None or len(path) == 0)):
            plan.planMore(numIters)
            path = plan.getPath()
            t1 = time.time()
        print(f"Planning time: {t1-t0} iterations{numIters}")
        #to be nice to the C++ module, do this to free up memory
        plan.space.close()
        plan.close()
        return path
    def assemble_result(self,plan):
        qlift = plan['lift']
        transfer = plan['transfer']
        qplace = plan['qplace']
        qpreplace = plan['qpreplace']
        retract = plan['retract']
        #TODO: construct the RobotTrajectory tuple (transfer,lower,retract)
        return (RobotTrajectory(self.robot,milestones=[self.qstart,qlift]+transfer),RobotTrajectory(self.robot,milestones=[qpreplace,qplace]),RobotTrajectory(self.robot,milestones=retract))

    def solve_item(self,plan,item):
        if item == 'lift':
            result = self.solve_lift()
            if result is None:
                print("Lifting object failed")
                return StepResult.FAIL,[]
            print("Found Lift config")
            return StepResult.CHILDREN,[result]
        if item == 'placement':
            print("Finding Placement")
            Ts = self.solve_placement()
            return StepResult.CHILDREN_AND_CONTINUE,Ts
        if item == 'qplace':
            print("Planning IK configuration")
            qplace = self.solve_qplace(plan['placement'])
            if qplace is None:
                print("IK solve failed... trying again")
                return StepResult.CONTINUE,[]
            else:
                print("IK solve succeeded, moving on to preplace planning")
                return StepResult.CHILDREN_AND_CONTINUE,[qplace]
        if item == 'qpreplace':
            qpreplace = self.solve_preplace(plan['qplace'])
            if qpreplace is None:
                print("Preplace planning Failed")
                return StepResult.FAIL,[]
            else:
                print("Preplace planning succeeded")
                return StepResult.CHILDREN,[qpreplace]
        if item == 'retract':
            retract = self.solve_retract(plan['qplace'])
            if retract is None:
                print("Retraction planning failed")
                return StepResult.FAIL,[]
            else:
                print("Retraction Planning Succeeded")
                return StepResult.CHILDREN,[retract]
        if item == 'transfer':
            transfer = self.solve_transfer(plan['qpreplace'], plan['lift'])
            if transfer is None:
                print("Transfer planning failed, trying again...")
                return StepResult.CONTINUE,[]
            else:
                print("Transfer planning succeeded!")
                return StepResult.CHILDREN,[transfer]

        raise ValueError("Invalid item "+item)


def plan_place(world,robot,obj,Tobject_gripper,gripper,goal_bounds):
    planner = PlacePlanner(world,robot,obj,Tobject_gripper,gripper,None,goal_bounds)
    time_limit = 60
    return planner.solve(time_limit)
def plan_place_target(world,robot,obj,Tobject_gripper,gripper,target):
    planner = PlacePlanner(world,robot,obj,Tobject_gripper,gripper,target)
    time_limit = 10
    return planner.solve(time_limit)