from klampt.plan import robotplanning
from klampt.plan.cspace import MotionPlan
from klampt.model import trajectory
from klampt import vis 
from klampt import RobotModel
import math
import time
import sys
sys.path.append("../motion/planners")
from motionGlobals import *

def feasible_plan(world,robot,qtarget):
    """Plans for some number of iterations from the robot's current configuration to
    configuration qtarget.  Returns the first path found.

    Returns None if no path was found, otherwise returns the plan.
    """
    moving_joints = [1,2,3,4,5,6,7]
    # space = robotplanning.makeSpace(world=world,robot=robot,edgeCheckResolution=1e-2,movingSubset=moving_joints)
    # plan = MotionPlan(space,type='prm')
    #TODO: maybe you should use planToConfig?
    planOpts = {'type':'sbl','perturbationRadius':0.5}
    plan = robotplanning.planToConfig(world, robot, qtarget, edgeCheckResolution=1e-2, 
                                        movingSubset=moving_joints, **planOpts)
    numIters = 80
    t1 = time.time()
    t0 = time.time()
    path = []
    c = 0
    while(t1-t0 < 10 and (path == None or len(path) == 0)):
        plan.planMore(numIters)
        path = plan.getPath()
        t1 = time.time()
        c +=1
    print(f"Planning time: {t1-t0} iterations: {numIters} looped times: {c}")
    #to be nice to the C++ module, do this to free up memory
    plan.space.close()
    plan.close()
    return path

def debug_plan_results(plan,robot):
    """Potentially useful for debugging planning results..."""
    assert isinstance(plan,MotionPlan)
    #this code just gives some debugging information. it may get expensive
    V,E = plan.getRoadmap()
    print(len(V),"feasible milestones sampled,",len(E),"edges connected")

    print("Plan stats:")
    pstats = plan.getStats()
    for k in sorted(pstats.keys()):
        print("  ",k,":",pstats[k])

    print("CSpace stats:")
    sstats = plan.space.getStats()
    for k in sorted(sstats.keys()):
        print("  ",k,":",sstats[k])
    """
    print("  Joint limit failures:")
    for i in range(robot.numLinks()):
        print("     ",robot.link(i).getName(),":",plan.space.ambientspace.joint_limit_failures[i])
    """

    path = plan.getPath()
    if path is None or len(path)==0:
        print("Failed to plan path between configuration")
        #debug some sampled configurations
        numconfigs = min(10,len(V))
        vis.debug("some milestones",V[2:numconfigs],world=world)
        pts = []
        for i,q in enumerate(V):
            robot.setConfig(q)
            pt = robot.link(9).getTransform()[1]
            pts.append(pt)
        for i,q in enumerate(V):
            vis.add("pt"+str(i),pts[i],hide_label=True,color=(1,1,0,0.75))
        for (a,b) in E:
            vis.add("edge_{}_{}".format(a,b),trajectory.Trajectory(milestones=[pts[a],pts[b]]),color=(1,0.5,0,0.5),width=1,pointSize=0,hide_label=True)
        return None

    print("Planned path with length",trajectory.RobotTrajectory(robot,milestones=path).length())
