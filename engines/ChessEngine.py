# from engines.EngineUtils import TileType,enum_to_piece

from klampt.math import vectorops,so3,se3
import sys
import math
import random

import chess
from chess import Piece,PieceType,Color

from klampt.robotsim import Mass

sys.path.append("../common")

# OBJECT_DIRECTORY = '../data/4d-Staunton_Full_Size_Chess_Set'
OBJECT_DIRECTORY = '../data/js4768'
TILE_DIRECTORY = '../data/objects/cube.off'

PIECE_SCALE = 0.05
TILE_SCALE = (0.05, 0.05, 0.005)

# WHITE_TILE = (0,0,0,1)
# BLACK_TILE = (1,1,1,1)
WHITE_TILE = (235/255, 235/255, 235/255, 1)
# BLACK_TILE = (125/255, 125/255, 125/255, 1)
BLACK_TILE = (36/255, 64/255, 51/255, 1)

TILE = 'tile'
PIECE = 'piece'
DEFAULT = 'default'

class ChessEngine:
    def __init__(self, world, tabletop, perspective_white=True):
        self.world = world
        self.tabletop = tabletop

        self.boardTiles = None
        self.pieces = None

        self.perspective_white = perspective_white

        self.WHITE = (253/255, 217/255, 168/255, 1)
        # self.BLACK = (45/255, 28/255, 12/255, 1)
        # self.BLACK = (110/255, 80/255, 16/255, 1)
        # self.BLACK = (79/255, 56/255, 6/255, 1)
        self.BLACK = (40/255, 40/255, 40/255, 1)

        self.board_rotation = 0

        self.pieceRotations = {}
        self.pieceRotations['N'] = math.pi/2
        self.pieceRotations['n'] = -math.pi/2 
        self.pieceRotations['K'] = math.pi/2
        self.pieceRotations['k'] = -math.pi/2
        self.pieceRotations['b'] = math.pi

    @classmethod
    def numberToPiece(cls, number):
        if number <= 0 or number > 12:
            return None 

        white = number <= 6 

        if not white:
            number -= 6 

        return Piece(number, white)

    @classmethod
    def pieceNameToPiece(cls, name):
        """ Takes the name of a chesspiece and returns the python-chess
        Piece type
        """
        split_name = name.split('_')

        black = split_name[1] == 'b'

        symbol = split_name[0][0]

        if split_name[0] == 'Knight':
            symbol = 'N'

        if black: 
            symbol = symbol.lower()

        return Piece.from_symbol(symbol)

    @classmethod
    def pieceNameToNumber(cls, name):
        """ Converts the chess piece name into a number
        """
        if name is None:
            return 0

        piece = ChessEngine.pieceNameToPiece(name)

        if piece.color:
            return piece.piece_type
        
        return piece.piece_type + 6


    def _getPieceRotation(self, name):
        """ Returns default rotation of a piece (default 0 for most pieces)
        """
        ptype = ChessEngine.pieceNameToPiece(name)

        if ptype.symbol() in self.pieceRotations:
            return self.pieceRotations[ptype.symbol()]
        else:
            return 0

    def _getPieceNumberAtTile(self, tilename):
        """
        Retrieves the file
        """
        pname, piece = self.boardTiles[tilename][PIECE] 

        if piece is None:
            return 0

        return ChessEngine.pieceNameToNumber(pname)

    def _clearBoard(self):
        """ Removes all current pieces from the board
        by setting their transform to identity and
        """
        R, t = se3.identity()

        for piece in self.pieces:
            self.pieces[piece].setTransform(R, t)

        for tile in self.boardTiles:
            self.boardTiles[tile][PIECE] = (None, None)

    def setPerspectiveWhite(self):
        return self.perspective_white

    def setPerspectiveWhite(self, perspective_white):
        self.perspective_white = perspective_white

    def randomizePieces(self, num_pieces=0):
        """ Randomizes placement of pieces WITHOUT ARRANGING
        THEM ONTO THE BOARD
        """
        self._clearBoard()

        if num_pieces <= 0:
            num_pieces = random.randint(1, len(self.pieces))

        sampled_pieces = random.sample(list(self.pieces), k=num_pieces)
        sampled_tiles = random.sample(list(self.boardTiles), k=num_pieces)

        for i in range(num_pieces):
            self.boardTiles[sampled_tiles[i]][PIECE] = (sampled_pieces[i], self.pieces[sampled_pieces[i]])
        
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

        for i,rank_name in enumerate(chess.RANK_NAMES):
            for j,file_name in enumerate(chess.FILE_NAMES):
                tilename = file_name + rank_name

                sx = (i - 4) * TILE_SCALE[0]
                sy = (j - 4) * TILE_SCALE[1]

                t = [
                    table_c_x + sx * math.cos(rotation) - sy * math.sin(rotation),
                    table_c_y + sy * math.cos(rotation) + sx * math.sin(rotation),
                    top_table_h + TILE_SCALE[2] / 2
                ]

                axis = [0,0,1]
                
                R = so3.from_axis_angle((axis, rotation))

                self.boardTiles[tilename][TILE].setTransform(R, t)
        
        self.board_rotation = rotation

    def arrangePieces(self, default=False, randomlyRotatePieces=False):
        """ Arranges pieces on a board. 

        :param: default if True, the pieces are arranged on the board
        as a default chess board setup 
        """
        for i,file_name in enumerate(chess.FILE_NAMES):
            for j,rank_name in enumerate(chess.RANK_NAMES):
                tilename = file_name + rank_name

                if default:
                    pname, piece = self.boardTiles[tilename][DEFAULT]
                else:
                    pname, piece = self.boardTiles[tilename][PIECE]

                tile = self.boardTiles[tilename][TILE]
                
                if piece is not None:
                    _, tile_T = tile.getTransform()

                    tile_bmin,tile_bmax = tile.geometry().getBBTight()

                    t = [ 
                        (tile_bmin[0] + tile_bmax[0]) / 2,
                        (tile_bmin[1] + tile_bmax[1]) / 2,
                        tile_T[2] + TILE_SCALE[2]
                    ]

                    axis = [0,0,1]

                    if randomlyRotatePieces:
                        rot = random.uniform(0, 2 * math.pi)
                    else:
                        rot = self.board_rotation + self._getPieceRotation(pname)

                    R = so3.from_axis_angle((axis, rot))

                    piece.setTransform(R, t)

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

            o.setName(oname)

            objs.append((oname, o))
            # self.world.add(oname, o)

            # TODO: Remove in later dates
            # Known bug that objects appear near the center of the world
            # All mashed together; ask Hauser about this
            o.setTransform(so3.identity(), [0,0,-1])

        return objs

    def loadBoard(self):
        """ Loads default configuration for pieces; loads tiles from
        object files; populates self.boardTiles
        """
        if self.boardTiles is not None:
            print('The board has already been loaded; subsequent calls to this function are redundant')
            return

        self.boardTiles = {}

        default_file = open('../engines/default_board.conf')

        default_pieces = {}

        for l in default_file.readlines():
            piece_info = l.strip().split(',')

            default_pieces[piece_info[0]] = piece_info[1] # self.pieces[piece_info[1]]

        default_file.close()

        white_tiles = self.loadObjects('Square', WHITE_TILE, 'w', 32, TILE_SCALE, False, TILE_DIRECTORY)
        black_tiles = self.loadObjects('Square', BLACK_TILE, 'b', 32, TILE_SCALE, False, TILE_DIRECTORY)

        white_idx = 0
        black_idx = 0

        for (i,file_name) in enumerate(chess.FILE_NAMES):
            for (j,rank_name) in enumerate(chess.RANK_NAMES):
                tilename = file_name + rank_name

                self.boardTiles[tilename] = {
                    TILE: None,
                    PIECE: (None, None),
                    DEFAULT: (None, None)
                }
                
                # Black tile
                if (i % 2 == 0 and j % 2 == 0) or (i % 2 != 0 and j % 2 != 0):
                    self.boardTiles[tilename][TILE] = black_tiles[black_idx][1]
                    black_idx += 1
                else: # White tile
                    self.boardTiles[tilename][TILE] = white_tiles[white_idx][1]
                    white_idx += 1

                if tilename in default_pieces:
                    self.boardTiles[tilename][PIECE] = (default_pieces[tilename], self.pieces[default_pieces[tilename]])
                    self.boardTiles[tilename][DEFAULT] = (default_pieces[tilename], self.pieces[default_pieces[tilename]])

    def getTableCenter(self):
        table_bmin,table_bmax = self.tabletop.geometry().getBBTight()

        return [
            (table_bmin[0] + table_bmax[0]) / 2,
            (table_bmin[1] + table_bmax[1]) / 2,
            table_bmax[2]
        ]

    def getBoardCorners(self):
        """
        Returns (clockwise) the set of points at the corners of the board
        
        NOTE: This method has undefined behavior if the board is not rotated by a degree divislbe by 90
        """
        [x,y,z] = self.getTableCenter()

        TILE_SPACE = 5

        rotation = self.board_rotation

        axis_dist = TILE_SPACE * TILE_SCALE[0]

        cos = math.cos(rotation)
        sin = math.sin(rotation)

        a8x = x + (axis_dist * cos - axis_dist * sin)
        a8y = y + (axis_dist * cos + axis_dist * sin) 

        h8x = x + (axis_dist * cos + axis_dist * sin)
        h8y = y + (-axis_dist * cos + axis_dist * sin) 

        h1x = x + (-axis_dist * cos + axis_dist * sin)
        h1y = y + (-axis_dist * cos - axis_dist * sin) 

        a1x = x + (-axis_dist * cos - axis_dist * sin)
        a1y = y + (axis_dist * cos - axis_dist * sin) 

        z += TILE_SCALE[2]

        corners = []

        if self.perspective_white:
            corners = [
                [a8x, a8y, z],
                [h8x, h8y, z],
                [h1x, h1y, z],
                [a1x, a1y, z]
            ]
        else:
            corners = [
                [h1x, h1y, z],
                [a1x, a1y, z],
                [a8x, a8y, z],
                [h8x, h8y, z]
            ]

        return corners

    def visualizeBoardCorners(self, vis):
        corner_coords = self.getBoardCorners()

        vis.add('corner0', corner_coords[0], scale=0.05, color=(1,0,0,1))
        vis.add('corner1', corner_coords[1], scale=0.05, color=(0,1,0,1))
        vis.add('corner2', corner_coords[2], scale=0.05, color=(0,0,1,1))
        vis.add('corner3', corner_coords[3], scale=0.05, color=(1,0,1,1))

    def getPieceArrangement(self):
        """
        Returns of values of PieceEnums in order that they appear in sensor image.

        NOTE: This is engineered specifically work with the DataGenerator; editor beware.
        
        TODO: Improve the documentation; account for rotation
        """
        arrangement = []

        if self.perspective_white:
            ranks = chess.RANK_NAMES[::-1]
            files = chess.FILE_NAMES
        else:
            ranks = chess.RANK_NAMES
            files = chess.FILE_NAMES[::-1]

        for rank_name in ranks:                 
            for file_name in files:
                tilename = file_name + rank_name
                arrangement.append(self._getPieceNumberAtTile(tilename))

        return arrangement

    def loadPieces(self):
        """ Loads pieces from object files, and populates self.pieces """
        if self.pieces is not None:
            print('Pieces have already been loaded; subsequent calls to this function are redundant')
            return 

        self.pieces = {}
        
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