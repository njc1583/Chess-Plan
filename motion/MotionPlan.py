
class MotionPlan:
    def __init__(self, world, robot, board):
        self.world = world
        self.robot = robot
        self.board = board      # boardTiles object from ChessEngine
    finger_pad_links = ['gripper:Link 4','gripper:Link 6']

    def is_collision_free_grasp(world,robot,object):
        #TODO: you might want to fix this to ignore collisions between finger pads and the object
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
        res = ik.solve(obj)
        if not res:
            return None
        return robot.getConfig()
    def is_collide():
        global c_world, c_robot, c_obj
        return is_collision_free_grasp(c_world, c_robot, c_obj)
    def plan_pick_one(world,robot,object,gripper,grasp):
        """
        Plans a picking motion for a given object and a specified grasp.

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
        global c_world, c_robot, c_obj
        c_world = world
        c_robot = robot
        c_obj = object
        qstart = robot.getConfig()
        grasp.ik_constraint.robot = robot  #this makes it more convenient to use the ik module
        
        #TODO solve the IK problem for qgrasp?
        #qgrasp = qstart
        solution = ik.solve_global(grasp.ik_constraint, iters=100, numRestarts = 10, feasibilityCheck=is_collide)
        if not solution:
            robot.setConfig(qstart)
            return None
        print(f"Solution: {solution}")
        qgrasp = robot.getConfig()
        qgrasp = grasp.set_finger_config(qgrasp)  #open the fingers the right amount
        qopen = gripper.set_finger_config(qgrasp,gripper.partway_open_config(1))   #open the fingers further
        distance = 0.2
        qpregrasp = retract(robot, gripper, vectorops.mul(gripper.primary_axis,-1*distance), local=True)   #TODO solve the retraction problem for qpregrasp?
        qstartopen = gripper.set_finger_config(qstart,gripper.partway_open_config(1))  #open the fingers of the start to match qpregrasp
        robot.setConfig(qstartopen)
        if qpregrasp is None:
            return None
        transit = feasible_plan(world,robot,qpregrasp)   #decide whether to use feasible_plan or optimizing_plan
        if not transit:
            return None

        #TODO: not a lot of collision checking going on either...
        for i in transit: #transit should be a list of configurations
            robot.setConfig(i)
            if not is_collide():
                return None
        robot.setConfig(qgrasp)
        qlift = retract(robot, gripper, vectorops.mul([0,0,1],distance), local=False) # move up a distance
        robot.setConfig(qstart)
        return (RobotTrajectory(robot,milestones=[qstart]+transit),RobotTrajectory(robot,milestones=[qpregrasp,qopen,qgrasp]),RobotTrajectory(robot,milestones=[qgrasp,qlift]))

        
    
