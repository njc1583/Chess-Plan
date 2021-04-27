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
import chess
import sys
sys.path.append('../common')
from known_grippers import robotiq_85_kinova_gen3
sys.path.append("../motion/planners")
from pickPlanner import *
from placePlanner import *
from motionGlobals import *
import grasp_database
PIECE_SCALE = 0.05
TILE_SCALE = (0.05, 0.05, 0.005)

class ChessMotion:
    """
        Executes pick-and-place motion plans on the chessboard
        Attributes:
            world (WorldModel): the world, containing robot, object, and other items that
                will need to be avoided.
            robot (RobotModel): the robot in its current configuration
            qstart(RobotConfig list): initial robot start config
            gripper (GripperInfo): the gripper
            boardTiles (dict): boardTiles object from ChessEngine
            currentObject (RigidObjectModel): the chess piece object to pick up currently
    """
    def __init__(self, world, robot, engine):#boardTiles):
        self.world = world
        self.robot = robot
        self.qstart = robot.getConfig()

        self.engine = engine
        self.currentObject = None
        self.currentMove = None

        self.executing_plan = False
        self.execute_start_time = time.time()
        self.solved_trajectory,self.trajectory_is_transfer = None,None

        self.gripper = robotiq_85_kinova_gen3
        self.Tobject_gripper = None
        self.db = grasp_database.GraspDatabase(self.gripper)
        if not self.db.load("../grasping/chess_grasps.json"):
            raise RuntimeError("Can't load grasp database?")
    def get_object_grasps(self, name, T_obj):
        """ Returns a list of transformed grasp objects from the db for the given object name and transform 
        """
        orig_grasps = self.db.object_to_grasps[name]
        grasps = [grasp.get_transformed(T_obj) for grasp in orig_grasps]
        return grasps
    def plan_to_piece(self,square):
        """ Finds the piece object on a given square and plans to pick it up
        """
        piece = self.engine.get_piece_obj_at(square)
        self.currentObject = piece
        name = piece.getName().split('_')[0]
        grasps = self.get_object_grasps(name, piece.getTransform())
        path = plan_pick_multistep(self.world,self.robot,self.currentObject,self.gripper,grasps)
        return path
    def plan_to_place(self,square:str):
        """ Before calling this function, 
        current robot config must be gripping the object so T_object_gripper will be correct
        """
        print(self.currentObject.getName())
        Tobj = self.currentObject.getTransform()
        link = self.robot.link(self.gripper.base_link)
        self.Tobject_gripper = se3.mul(se3.inv(link.getTransform()),Tobj)
        T_target = self.engine.get_square_transform(square, self.currentObject.getName())
        print(T_target)
        path = plan_place_target(self.world, self.robot,self.currentObject,self.Tobject_gripper,self.gripper,T_target)
        self.currentObject.setTransform(*Tobj)
        return path
    def make_move(self, san: str):
        """ Constructs the trajectory for a chess move string in standard algebraic notation
            and triggers execution.
            Returns trajectory if move was legal and a path could be found, else None
        """
        self.currentMove,start_square,target_square = self.engine.check_move(san)
        if self.currentMove == None:
            return None,None
        self.robot.setConfig(self.qstart)
        path = self.plan_to_piece(start_square)
        solved_trajectory = None
        trajectory_is_transfer = None
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
            self.robot.setConfig(approach.milestones[-1])
            tTarget = self.engine.get_square_transform(target_square, self.currentObject.getName())
            vis.add("targetTransform", tTarget)
            print("attempting plan to place")
            res = self.plan_to_place(target_square)
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
        self.robot.setConfig(self.qstart)
        if solved_trajectory is not None:
            self.executing_plan = True
            self.execute_start_time = time.time()
        return solved_trajectory, trajectory_is_transfer
    def loop_callback(self):
        if not self.executing_plan:
            san = input("Enter Chess Move:")
            self.solved_trajectory,self.trajectory_is_transfer = self.make_move(san)
            print("Made Plan at:", self.execute_start_time)
            return
        if self.solved_trajectory:
            t = time.time()-self.execute_start_time
            vis.addText("time","Time %.3f"%(t),position=(10,10))
            qcurrent = self.solved_trajectory.eval(t)
            self.robot.setConfig(qcurrent)
            during_transfer = self.trajectory_is_transfer.eval(t)[0]
            if during_transfer:
                self.currentObject.setTransform(*se3.mul(self.robot.link(9).getTransform(),self.Tobject_gripper))
            if t > self.solved_trajectory.duration():
                self.executing_plan = False
                self.solved_trajectory = None
                self.trajectory_is_transfer = None
                # Update move made on chessBoard and boardTiles
                self.engine.update_board(self.currentMove)
                self.currentObject = None
                self.currentMove = None
                self.robot.setConfig(self.qstart)

    def check_collision(self):
        return is_collision_free_grasp(self.world, self.robot, self.currentObject)
    def go_to_square(self,square):
        Tobj = self.currentObject.getTransform()
        link = self.robot.link(self.gripper.base_link)
        Tobject_gripper = se3.mul(se3.inv(link.getTransform()),Tobj)
        T_target = self.engine.get_square_transform(square)
        T_grip = se3.mul(T_target,se3.inv(Tobject_gripper))
        return self.solve_robot_ik(T_grip)
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
