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

import cv2

import DataUtils

from klampt.robotsim import RigidObjectModel
sys.path.append("../common")
sys.path.append("../engines")

from ChessEngine import ChessEngine

DIST_FROM_BOARD = 0.5

class DataGenerator:
    def __init__(self):
        self._loadWorld('./main.xml')
        self._loadSensor('./camera.rob')

        self.chessEngine = ChessEngine(self.world, self.world.terrain('tabletop'))
        self.chessEngine.loadPieces()
        self.chessEngine.loadBoard()

        self.chessEngine.arrangeBoard(90)
        self.chessEngine.arrangePieces()

        self._imageDir = 'image_dataset'

        self._colorDir = self._imageDir + '/color'
        self._depthDir = self._imageDir + '/depth'
        self._rectifiedDir = self._imageDir + '/rectified'

        self._metaDataFN = self._imageDir + '/metadata.csv'

        self._dirs = [self._imageDir, self._colorDir, self._depthDir, self._rectifiedDir]

        self._colorFNFormat = self._colorDir + '/%06d_%02d.png'
        self._depthFNFormat = self._depthDir + '/%06d_%02d.png'
        self._rectifiedFNFormat = self._rectifiedDir + '/%06d.png'

        self._rectifiedPictureCorners = np.float32([
            [DataUtils.IMAGE_SIZE,0],
            [0,0],
            [0,DataUtils.IMAGE_SIZE],
            [DataUtils.IMAGE_SIZE,DataUtils.IMAGE_SIZE]
        ])
        self._boardWorldCorners = self.chessEngine.getBoardCorners()

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
        for d in self._dirs:
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

    def generateImages(self, max_pics=100, data_distribution=None):
        assert max_pics > 0
        
        if data_distribution is None:
            data_distribution = DataUtils.LIMITED_DISTRIBUTION
        
        assert len(data_distribution) == 13

        self._createDatasetDirectory()

        for i in range(self.world.numRigidObjects()):
            self.world.rigidObject(i).appearance().setSilhouette(0)

        metadata_f = open(self._metaDataFN, mode='w+')
        metadata_f.write('color,depth,piece\n')

        DEPTH_SCALE = 8000

        def loop_through_sensors(world=self.world, sensor=self.sensor, max_pics=max_pics):
            for counter in tqdm(range(max_pics)):
                if counter > 0:
                    self.chessEngine.randomizePieces(32)

                # Arrange pieces and model world
                self.chessEngine.arrangePieces()
                
                self._randomlyRotateCamera(20, 40, 0.6)

                sensor.kinematicReset()
                sensor.kinematicSimulate(world, 0.01)

                rgb, depth = sensing.camera_to_images(self.sensor)

                # Project RGB and depth images to rectify them
                local_corner_coords = np.float32([sensing.camera_project(self.sensor, self.robot, pt)[:2] for pt in self._boardWorldCorners])

                H = cv2.getPerspectiveTransform(local_corner_coords, self._rectifiedPictureCorners)

                color_rectified = cv2.warpPerspective(rgb, H, (DataUtils.IMAGE_SIZE, DataUtils.IMAGE_SIZE))
                depth_rectified = cv2.warpPerspective(depth, H, (DataUtils.IMAGE_SIZE, DataUtils.IMAGE_SIZE))

                depth_rectified = np.uint8((depth_rectified - depth_rectified.min()) / (depth_rectified.max() - depth_rectified.min()))

                pieces_arrangement = self.chessEngine.getPieceArrangement()

                images, classes = DataUtils.split_image_by_classes(color_rectified, depth_rectified, data_distribution, pieces_arrangement)

                rectified_fname = self._rectifiedFNFormat % (counter)
                Image.fromarray(color_rectified).save(rectified_fname)

                for idx in range(sum(data_distribution)):
                    color_fname = self._colorFNFormat % (counter, idx)
                    depth_fname = self._depthFNFormat % (counter, idx)

                    Image.fromarray(images[idx,:,:,:3]).save(color_fname)
                    Image.fromarray(images[idx,:,:,3]).save(depth_fname)

                    metadata_f.write(f'{color_fname},{depth_fname},{classes[idx]}\n')

            vis.show(False)

        vis.loop(callback=loop_through_sensors)
        
        metadata_f.close()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_images', '-n', help='number of generated images', default=100, type=int)
    parser.add_argument('--delete_dataset', '-dd', action='store_true', help='delete dataset before processing images')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    data_generator = DataGenerator()

    if args.delete_dataset:
        data_generator.deleteDataset()

    data_generator.generateImages(args.num_images, DataUtils.LIMITED_DISTRIBUTION)