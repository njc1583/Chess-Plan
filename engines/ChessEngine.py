from klampt.math import vectorops,so3,se3
import sys
import math

from klampt.robotsim import Mass
sys.path.append("../common")

# OBJECT_DIRECTORY = '../data/4d-Staunton_Full_Size_Chess_Set'
OBJECT_DIRECTORY = '../data/js4768'
TILE_DIRECTORY = '../data/objects/cube.off'

PIECE_SCALE = 0.05
TILE_SCALE = (0.05, 0.05, 0.005)

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

        self.board_rotation = 0

    def arrangeBoard(self, rotation=0):
        """ Arranges chessboard on the tabletop

        :param: rotation, in degrees, the clockwise rotation of the 
        chessboard
        """
        rotation = math.radians(rotation)

        table_bmin,table_bmax = self.tabletop.geometry().getBBTight()
        
        top_table_h = table_bmax[2]

        table_c_x = (table_bmax[0] + table_bmin[0]) / 2
        table_c_y = (table_bmax[1] + table_bmin[1]) / 2

        for i in range(len(BOARD_X)):
            for j in range(len(BOARD_Y)):
                tilename = BOARD_X[i] + BOARD_Y[j]

                sx = (i - 4) * TILE_SCALE[0]
                sy = (j - 4) * TILE_SCALE[1]

                t = [
                    table_c_x + sx * math.cos(rotation) - sy * math.sin(rotation),
                    table_c_y + sy * math.cos(rotation) + sx * math.sin(rotation),
                    top_table_h + TILE_SCALE[2] / 2
                ]

                axis = [0,0,1]
                
                R = so3.from_axis_angle((axis, rotation))

                self.boardTiles[tilename]['tile'].setTransform(R, t)
        
        self.board_rotation = rotation

    def arrangePieces(self, default=False):
        """ Arranges pieces on a board. 

        :param: default if True, the pieces are arranged on the board
        as a default chess board setup 
        """
        for i in range(len(BOARD_X)):
            for j in range(len(BOARD_Y)):
                tilename = BOARD_X[i] + BOARD_Y[j]

                if default:
                    tile = self.boardTiles[tilename]['default']
                else:
                    tile = self.boardTiles[tilename]['tile']
                
                piece = self.boardTiles[tilename]['piece']

                if piece is not None:
                    tile_R, tile_T = tile.getTransform()

                    tile_bmin,tile_bmax = tile.geometry().getBBTight()

                    t = [ 
                        (tile_bmin[0] + tile_bmax[0]) / 2,
                        (tile_bmin[1] + tile_bmax[1]) / 2,
                        tile_T[2] + TILE_SCALE[2]
                    ]

                    piece.setTransform(so3.identity(), t)

    def loadObjects(self, name, color, colorn, amount, scale, piece, fname=None):
        """ Loads an object from a file and sets scale and color of the
        object; gives it a name; and sets the mass of an object
        """
        if fname is None:
            fname = f'{OBJECT_DIRECTORY}/{name}.stl'

        objs = []

        for i in range(amount):
            o = self.world.loadRigidObject(fname)

            if piece:
                o.geometry().scale(scale)
            else:
                sx,sy,sz = scale
                o.geometry().scale(sx,sy,sz)

            m = Mass()
            m.estimate(o.geometry(), 0.25)
            o.setMass(m)

            r, g, b, a = color 

            o.appearance().setColor(r,g,b,a)
            oname = f'{name}_{colorn}_{i}'

            objs.append((oname, o))
            self.world.add(oname, o)

        return objs

    def loadBoard(self):
        """ Loads default configuration for pieces; loads tiles from
        object files; populates self.boardTiles
        """
        default_file = open('../engines/default_board.conf')

        default_pieces = {}

        for l in default_file.readlines():
            piece_info = l.strip().split(',')

            default_pieces[piece_info[0]] = self.pieces[piece_info[1]]

        default_file.close()

        white_tiles = self.loadObjects('Square', (1,1,1,1), 'w', 32, TILE_SCALE, False, TILE_DIRECTORY)
        black_tiles = self.loadObjects('Square', (0,0,0,1), 'b', 32, TILE_SCALE, False, TILE_DIRECTORY)

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
                    self.boardTiles[tilename]['tile'] = black_tiles[black_idx][1]
                    black_idx += 1
                else: # White tile
                    self.boardTiles[tilename]['tile'] = white_tiles[white_idx][1]
                    white_idx += 1

                if tilename in default_pieces:
                    self.boardTiles[tilename]['piece'] = default_pieces[tilename]
                    self.boardTiles[tilename]['default'] = default_pieces[tilename]

    def loadPieces(self):
        """ Loads pieces from object files, and populates self.pieces """
        pieces = [] 
        
        pieces.extend(self.loadObjects('Rook', self.WHITE, 'w', 2, PIECE_SCALE, True))
        pieces.extend(self.loadObjects('Rook', self.BLACK, 'b', 2, PIECE_SCALE, True))
        
        pieces.extend(self.loadObjects('Bishop', self.WHITE, 'w', 2, PIECE_SCALE, True))
        pieces.extend(self.loadObjects('Bishop', self.BLACK, 'b', 2, PIECE_SCALE, True))
        
        pieces.extend(self.loadObjects('Knight', self.WHITE, 'w', 2, PIECE_SCALE, True))
        pieces.extend(self.loadObjects('Knight', self.BLACK, 'b', 2, PIECE_SCALE, True))

        pieces.extend(self.loadObjects('King', self.WHITE, 'w', 1, PIECE_SCALE, True))
        pieces.extend(self.loadObjects('King', self.BLACK, 'b', 1, PIECE_SCALE, True))

        pieces.extend(self.loadObjects('Queen', self.WHITE, 'w', 1, PIECE_SCALE, True))
        pieces.extend(self.loadObjects('Queen', self.BLACK, 'b', 1, PIECE_SCALE, True))

        pieces.extend(self.loadObjects('Pawn', self.WHITE, 'w', 8, PIECE_SCALE, True))
        pieces.extend(self.loadObjects('Pawn', self.BLACK, 'b', 8, PIECE_SCALE, True))

        for (piece_name, piece_obj) in pieces:
            self.pieces[piece_name] = piece_obj