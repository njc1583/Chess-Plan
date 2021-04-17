from klampt.robotsim import Mass
from klampt.math import vectorops,so3,se3

OBJECT_DIRECTORY = '../data/js4768'
TILE_DIRECTORY = '../data/objects/cube.off'
PIECE_SCALE = 0.05
TILE_SCALE = (0.05, 0.05, 0.005)
TILE = 'tile'
PIECE = 'piece'
WHITE = (253/255, 217/255, 168/255, 1)
def loadPiece(world, name):
    if name == TILE:
        return loadObjects(world, name, WHITE, 'w', 1, TILE_SCALE, False, TILE_DIRECTORY)
    return loadObjects(world, name, WHITE, 'w', 1, PIECE_SCALE, True)
def loadObjects(world, name, color, colorn, amount, scale, piece, fname=None):
        """ Loads an object from a file and sets scale and color of the
        object; gives it a name; and sets the mass of an object
        """
        if fname is None:
            fname = f'{OBJECT_DIRECTORY}/{name}.stl'

        objs = []

        for i in range(amount):
            o = world.loadRigidObject(fname)

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
            o.setTransform(so3.identity(), [0,0,0])

        return objs