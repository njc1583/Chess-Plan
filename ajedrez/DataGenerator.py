from klampt import WorldModel,Simulator
from klampt.io import resource
from klampt import vis 
from klampt.model.trajectory import Trajectory,RobotTrajectory,path_to_trajectory
from klampt.math import vectorops,so3,se3
from klampt.model import sensing

from PIL import Image

from tqdm import tqdm

import numpy as np
import math
import random
import sys
import os
import argparse
import shutil

from klampt.robotsim import RigidObjectModel
sys.path.append("../common")
sys.path.append("../engines")

from ChessEngine import ChessEngine
from PieceEnum import TileType

DIST_FROM_BOARD = 0.5

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

        self._metaDataFN = self._imageDir + '/metadata.csv'

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

    def deleteDataset(self):
        try:
            shutil.rmtree(self._imageDir)
        except OSError as e:
            print(f'Error deleting {self._imageDir}: {e}')

    def _randomlyRotateCamera(self, min_r=0, max_r=70, dist=DIST_FROM_BOARD):
        """
        Randomly rotates camera about x-axis and zooms out from the center of the 
        tabletop

        :param: min_r: the minimum random rotation in degrees
        :param: max_r: the maximum random rotation in degrees
        :param: dist: the distance to zoom out from center of the table 

        :return: the angle of rotation sampled
        """
        min_r = math.radians(min_r)
        max_r = math.radians(max_r)

        table_center = self.chessEngine.getTableCenter()

        table_R = so3.from_axis_angle(([1,0,0], -math.pi))
        table_t = table_center

        rot_deg = random.uniform(min_r, max_r)
        zoom_out_R = so3.from_axis_angle(([1,0,0], rot_deg))
        zoom_out_t = vectorops.mul([0,math.sin(rot_deg),-math.cos(rot_deg)], dist)

        xform = se3.mul((table_R, table_t), (zoom_out_R, zoom_out_t))

        sensing.set_sensor_xform(self.sensor, xform)

        return rot_deg

    def _rotateCamera(self, r, dist=DIST_FROM_BOARD):
        """
        Rotates a camera and zooms out from center of the tabletop

        :param: r: rotation in degrees
        :param: dist: distance to zoom out from center of the table

        :return: angle of rotation
        """
        return self._randomlyRotateCamera(r, r, dist)

    def generateImages(self, max_pics=100, save_depth=True):
        self._createDatasetDirectory()

        for i in range(self.world.numRigidObjects()):
            self.world.rigidObject(i).appearance().setSilhouette(0)

        metadata_f = open(self._metaDataFN, mode='w+')
        metadata_f.write('color,depth,pieces\n')

        def loop_through_sensors(world=self.world, sensor=self.sensor, max_pics=max_pics, save_depth=save_depth):

            depth_scale = 8000

            for counter in tqdm(range(max_pics)):
                if counter > 0:
                    self.chessEngine.randomizePieces()

                self.chessEngine.arrangePieces()
                
                self._randomlyRotateCamera(20, 40)

                sensor.kinematicReset()
                sensor.kinematicSimulate(world, 0.01)

                rgb,depth = sensing.camera_to_images(self.sensor)

                Image.fromarray(rgb).save(self._colorFNFormat % counter)

                pieces_arrangement = self.chessEngine.getPieceArrangement()
                metadata_f.write(f'{self._colorFNFormat % counter},{self._depthFNFormat % counter},{pieces_arrangement}\n')

                if save_depth:
                    depth_quantized = (depth * depth_scale).astype(np.uint32)
                    Image.fromarray(depth_quantized).save(self._depthFNFormat % counter)

            vis.show(False)

        vis.loop(callback=loop_through_sensors)
        
        metadata_f.close()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_images', '-n', help='number of generated images', default=100, type=int)
    parser.add_argument('--save_depth', '-sd', action='store_true', help='save depth images')
    parser.add_argument('--delete_dataset', '-dd', action='store_true', help='delete dataset before processing images')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    data_generator = DataGenerator()

    if args.delete_dataset:
        data_generator.deleteDataset()

    data_generator.generateImages(args.num_images, save_depth=args.save_depth)