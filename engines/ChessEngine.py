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

    def arrangeBoard(self, default=False):
        table_bmin,table_bmax = self.tabletop.geometry().getBBTight()
        tile_bmin,tile_bmax = self.boardTiles['A1']['tile'].geometry().getBBTight()
        
        top_table_h = table_bmax[2]
        tile_height = tile_bmax[2] - tile_bmin[2]

        tile_x = tile_bmax[0] - tile_bmin[0]
        tile_y = tile_bmax[1] - tile_bmin[1]

        table_c_x = table_bmin[0] + (table_bmax[0] - table_bmin[0]) / 2
        table_c_y = table_bmin[1] + (table_bmax[1] - table_bmin[1]) / 2

        start_x = table_c_x - 4 * tile_x 
        start_y = table_c_y - 4 * tile_y

        print(table_bmin, table_bmax, tile_bmin, tile_bmax)

        for i in range(len(BOARD_X)):
            for j in range(len(BOARD_Y)):
                tilename = BOARD_X[i] + BOARD_Y[j]

                t = [
                    start_x + i * tile_x,
                    start_y + j * tile_y,
                    top_table_h + tile_height / 2
                ]

                self.boardTiles[tilename]['tile'].setTransform(so3.identity(), t)

                piece = self.boardTiles[tilename]['piece']

                if piece is not None:
                    piece_bmin,piece_bmax = piece.geometry().getBBTight()
                    com = piece.getMass().getCom()

                    piece_h = com[2] -  piece_bmin[2]

                    tile_com = self.boardTiles[tilename]['tile'].getMass().getCom()

                    t = [tile_com[0], tile_com[1], piece_h]

                    piece.setTransform(so3.identity(), t)


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

            default_pieces[piece_info[0]] = self.pieces[piece_info[1]]

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
                    self.boardTiles[tilename]['tile'] = white_tiles[white_idx][1]
                    white_idx += 1
                else: # White tile
                    self.boardTiles[tilename]['tile'] = black_tiles[black_idx][1]
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