from klampt import WorldModel,Simulator
from klampt.io import resource
from klampt import vis 
from klampt.model.trajectory import Trajectory,RobotTrajectory,path_to_trajectory
from klampt.math import vectorops,so3,se3
from klampt.model import sensing

from PIL import Image

import math
import random
import time
import sys

from klampt.robotsim import RigidObjectModel
sys.path.append("../common")
sys.path.append("../engines")

from ChessEngine import ChessEngine

class DataGenerator:
    def __init__(self):
        self._loadWorld('./main.xml')
        self._loadSensor('./camera.rob')

        self.chessEngine = ChessEngine(self.world, self.world.terrain('tabletop'))
        self.chessEngine.loadPieces()
        self.chessEngine.loadBoard()

        self.chessEngine.arrangeBoard()
        self.chessEngine.arrangePieces()

    def _loadWorld(self, world_fn):
        world = WorldModel()
        res = world.readFile(world_fn)
    
        if not res:
            print("Unable to read file",world_fn)
            exit(0)
        for i in range(world.numRigidObjects()):
            obj = world.rigidObject(i)
            #this will perform a reasonable center of mass / inertia estimate
            m = obj.getMass()
            m.estimate(obj.geometry(),mass=0.454,surfaceFraction=0.2)
            obj.setMass(m)

        self.world = world

    def _loadSensor(self, sensor_fn):
        try:
            self.world.readFile(sensor_fn)
        except Exception as e:
            print(f'Error in loading filename: {e}')
        
        sensor = self.world.robot(0).sensor(0)

        if sensor is None:
            print("Sensor is none: check sensor filename")
            exit(1)

        self.sensor = sensor
    
    def generateImages(self):
        self.sensor.kinematicReset()
        self.sensor.kinematicSimulate(self.world, 0.01)

        # t = [0,0,0]
        # R = so3.from_axis_angle(([1,0,0], math.pi/4))

        # sensing.set_sensor_xform(self.sensor, (R,t))

        rgb,depth = sensing.camera_to_images(self.sensor)

        Image.fromarray(rgb).save('./test.png')

if __name__ == "__main__":
    data_generator = DataGenerator()
    data_generator.generateImages()