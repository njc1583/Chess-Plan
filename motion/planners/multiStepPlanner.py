import math
import time

class StepResult:    
    FAIL = 0
    COMPLETE = 1
    CONTINUE = 2
    CHILDREN = 3
    CHILDREN_AND_CONTINUE = 4

class PQNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def __lt__(self, other):
        return self.value < other.value

    def __str__(self):
        return str("{} : {}".format(self.key, self.value))

class MultiStepPlanner:
    """A generic multi step planner that can be subclassed to implement multi-step planning
    behavior.

    Subclass will need to:
        - implement choose_item() or provide a sequence of items to solve to the initializer.
        - implement solve_item() to complete items of the plan
        - implement score() for a custom heuristic (lower scores are better)
    
    """
    def __init__(self,items=None):
        self.W = [PQNode({},0)]
        if items is not None:
            self.items = items
        else:
            self.items = []
        self.pending_solutions = []

    def choose_item(self,plan):
        """Returns an item that is not yet complete in plan.  Default loops through items
        provided in the constructor (fixed order), but a subclass can do more sophisticated
        reasoning.

        Args:
            plan (dict): a partial plan.
        """
        for i in self.items:
            if i not in plan:
                return i
        return None

    def on_solve(self,plan,item,soln):
        """Callback that can do logging, etc."""
        pass

    def solve_item(self,plan,item):
        """Attempts to solve the given item in the plan.  Can tell the high-level planner to
        stop this line of reasoning (FAIL), the solution(s) completes the plan (COMPLETE), this item
        needs more time to solve (CONTINUE), several options have been generated and we wish
        stop planning this item (CHILDREN), or options have been generated and we wish to
        continue trying to generate more (CHILDREN_AND_CONTINUE).

        If you want to cache something for this item for future calls, put it under plan['_'+item].

        Args:
            plan (dict): a partial plan
            item (str): an item to solve.

        Returns:
            tuple: a pair (status,children) where status is one of the StepResult codes
            FAIL, COMPLETE, CONTINUE, CHILDREN, CHILDREN_AND_CONTINUE, and
            children is a list of solutions that complete more of the plan.
        """
        return StepResult.FAIL,[]
    
    def score(self,plan):
        """Returns a numeric score for the plan.  Default checks the score"""
        num_solved = len([k for k in plan if not k.startswith('_')])
        return plan.get('_solve_time',0) - 0.2*num_solved

    def assemble_result(self,plan):
        """Turns a dict partial plan into a complete result.  Default just
        returns the plan dict.
        """
        return dict((k,v) for (k,v) in plan.items() if not k.startwith('_'))

    def solve(self,tmax=float('inf')):
        """Solves the whole plan using least-commitment planning. 

        Can be called multiple times to produce multiple solutions.
        """
        import heapq,copy
        if self.pending_solutions:
            soln = self.pending_solutions.pop(0)
            return self.assemble_result(soln)
        tstart = time.time()
        while len(self.W)>0:
            if time.time()-tstart > tmax:
                return None
            node = heapq.heappop(self.W)
            plan = node.key
            prio = node.value
            item = self.choose_item(plan)
            if item is None:
                #no items left, done
                return self.assemble_result(plan)
            #print("Choosing item",item,"priority",prio)
            t0 = time.time()
            status,children = self.solve_item(plan,item)
            t1 = time.time()
            plan.setdefault('_solve_time',0)
            plan['_solve_time'] += t1-t0
            if status == StepResult.FAIL:
                continue
            elif status == StepResult.COMPLETE:
                assert len(children) > 0,"COMPLETE was returned but without an item solution?"
                soln = children[0]
                self.on_solve(plan,item,soln)
                plan[item] = soln
                for soln in children[1:]:
                    self.on_solve(plan,item,soln)
                    child = copy.copy(plan)
                    child[item] = soln
                    self.pending_solutions.append(child)
                return self.assemble_result(plan)
            else:
                if status == StepResult.CHILDREN_AND_CONTINUE or status == StepResult.CONTINUE:
                    heapq.heappush(self.W,PQNode(plan,self.score(plan)))
                for soln in children:
                    self.on_solve(plan,item,soln)
                    child = copy.copy(plan)
                    child[item] = soln
                    child['_solve_time'] = 0
                    heapq.heappush(self.W,PQNode(child,self.score(child)))
                    #print("Child priority",self.score(child))
        return None
