
class Point(object):
    def __init__(self, x=0, y=0):
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    def distance(p1, p2):
        return ((p1.x-p2.x)**2+(p1.y-p2.y)**2)**0.5

    def __str__(self):
        return '('+str(self.x)+','+str(self.y)+')'
