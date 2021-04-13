from klampt import WorldModel,Simulator
from klampt.io import resource
from klampt import vis 
from klampt.model.trajectory import Trajectory,RobotTrajectory,path_to_trajectory
from klampt.math import vectorops,so3,se3
from klampt.model import sensing

from PIL import Image

import numpy as np
import math
import random
import time
import sys
import os

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

        self._imageDir = 'image_dataset'
        self._colorDir = 'image_dataset/color'
        self._depthDir = 'image_dataset/depth'

        self._colorFNFormat = self._colorDir + '/%06d.png'
        self._depthFNFormat = self._depthDir + '/%06d.png' 

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
        
        robot = self.world.robot(0)

        if robot is None:
            print("Robot is none: check sensor filename")
            exit(1)

        sensor = self.world.robot(0).sensor(0)

        if sensor is None:
            print("Sensor is none: check sensor filename")
            exit(1)

        self.robot = robot
        self.sensor = sensor
    
    def _createDatasetDirectory(self):
        dirs = [self._imageDir, self._colorDir, self._depthDir]

        for d in dirs:
            try:
                os.mkdir(d)
            except Exception:
                pass 

    def _randomlyRotateCamera(self, min_r=0, max_r=70):
        min_r = math.radians(min_r)
        max_r = math.radians(max_r)

        table_center = self.chessEngine.getTableCenter()

        table_R = so3.from_axis_angle(([1,0,0], -math.pi))
        table_t = table_center

        rot_deg = random.uniform(min_r, max_r)
        zoom_out_R = so3.from_axis_angle(([1,0,0], rot_deg))
        zoom_out_t = vectorops.mul([0,math.sin(rot_deg),-math.cos(rot_deg)], 0.5)

        xform = se3.mul((table_R, table_t), (zoom_out_R, zoom_out_t))

        sensing.set_sensor_xform(self.sensor, xform)

        return rot_deg

    def _rotateCamera(self, r):
        return self._randomlyRotateCamera(r, r)

    def generateImages(self, max_pics=100):
        self._createDatasetDirectory()

        for i in range(self.world.numRigidObjects()):
            self.world.rigidObject(i).appearance().setSilhouette(0)

        self._randomlyRotateCamera()

        def loop_through_sensors(world=self.world, sensor=self.sensor, max_pics=max_pics):

            depth_scale = 8000

            self._rotateCamera(45)

            for counter in range(max_pics):
                if counter > 0:
                    self.chessEngine.randomizePieces()

                self.chessEngine.arrangePieces()
                
                sensor.kinematicReset()
                sensor.kinematicSimulate(world, 0.01)

                rgb,depth = sensing.camera_to_images(self.sensor)

                Image.fromarray(rgb).save(self._colorFNFormat%counter)

                depth_quantized = (depth * depth_scale).astype(np.uint32)

                Image.fromarray(depth_quantized).save(self._depthFNFormat%counter)

            vis.show(False)

        vis.loop(callback=loop_through_sensors)

if __name__ == "__main__":
    data_generator = DataGenerator()
    data_generator.generateImages(10)