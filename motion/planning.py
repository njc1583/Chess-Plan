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