from klampt import WorldModel,Simulator
from klampt.io import resource
from klampt import vis 
from klampt.model.trajectory import Trajectory,RobotTrajectory,path_to_trajectory
from klampt.math import vectorops,so3,se3
import math
import random
import time
import sys

from klampt.robotsim import RigidObjectModel,Mass
sys.path.append("../common")

OBJECT_DIRECTORY = '../data/4d-Staunton_Full_Size_Chess_Set'

BOARD_X = 'ABCDEFGH'
BOARD_Y = '12345678'

class ChessEngine:
    def __init__(self, world, tabletop):
        self.world = world
        self.tabletop = tabletop

        self.boardTiles = {}
        self.pieces = {} 

        self.WHITE = (253/255, 217/255, 168/255, 1)
        self.BLACK = (45/255, 28/255, 12/255, 1)

    def loadPiece(self, name, color, colorn, amount):
        scale = 0.001
        
        fname = f'{OBJECT_DIRECTORY}/{name}.stl'

        objs = []

        for i in range(amount):
            o = self.world.loadRigidObject(fname)
            o.geometry().scale(scale)

            # m = Mass()
            # m.estimate(o.geometry(), 0.25)
            # o.setMass(m)

            r, g, b, a = color 

            o.appearance().setColor(r,g,b,a)
            oname = f'{name}_{colorn}_{i}'

            objs.append((oname, o))
            self.world.add(oname, o)

        return objs

    def loadBoard(self):
        # Load default values
        default_file = open('../engines/default_board.conf')

        default_pieces = {}

        for l in default_file.readlines():
            piece_info = l.strip().split(',')

            default_pieces[piece_info[0]] = piece_info[1]

        default_file.close()

        white_tiles = self.loadPiece('Square', (1,1,1,1), 'w', 32)
        black_tiles = self.loadPiece('Square', (0,0,0,1), 'b', 32)

        white_idx = 0
        black_idx = 0

        for i in range(len(BOARD_X)):
            for j in range(len(BOARD_Y)):
                tilename = BOARD_X[i] + BOARD_Y[j]

                self.boardTiles[tilename] = {
                    'tile': None,
                    'piece': None,
                    'default': None
                }
                
                # Black tile
                if (i % 2 == 0 and j % 2 == 0) or (i % 2 != 0 and j % 2 != 0):
                    self.boardTiles[tilename]['tile'] = white_tiles[white_idx]
                    white_idx += 1
                else: # White tile
                    self.boardTiles[tilename]['tile'] = black_tiles[black_idx]
                    black_idx += 1

                if tilename in default_pieces:
                    self.boardTiles[tilename]['piece'] = default_pieces[tilename]
                    self.boardTiles[tilename]['default'] = default_pieces[tilename]

    def loadPieces(self):
        pieces = [] 
        
        pieces.extend(self.loadPiece('Rook', self.WHITE, 'w', 2))
        pieces.extend(self.loadPiece('Rook', self.BLACK, 'b', 2))
        
        pieces.extend(self.loadPiece('Bishop', self.WHITE, 'w', 2))
        pieces.extend(self.loadPiece('Bishop', self.BLACK, 'b', 2))
        
        pieces.extend(self.loadPiece('Knight', self.WHITE, 'w', 2))
        pieces.extend(self.loadPiece('Knight', self.BLACK, 'b', 2))

        pieces.extend(self.loadPiece('King', self.WHITE, 'w', 1))
        pieces.extend(self.loadPiece('King', self.BLACK, 'b', 1))

        pieces.extend(self.loadPiece('Queen', self.WHITE, 'w', 1))
        pieces.extend(self.loadPiece('Queen', self.BLACK, 'b', 1))

        pieces.extend(self.loadPiece('Pawn', self.WHITE, 'w', 8))
        pieces.extend(self.loadPiece('Pawn', self.BLACK, 'b', 8))

        for (piece_name, piece_obj) in pieces:
            self.pieces[piece_name] = piece_obj