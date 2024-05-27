import numpy as np
import math
import pdb

from ilqr.Point import Point


class Boundaries:
    def __init__(self, args, track_id, polylines: list[Point], valid_points: list[Point]):
        # valid_points is the points NOT voliate the constrains
        self.args = args
        self.track_id = track_id
        self.polylines = polylines
        self.num_of_points = len(polylines)
        self.valid_points = valid_points
        self.inequality_constrains = []
        print("polylines:")
        for i in range(0, len(polylines)):
            print('Point x: ', polylines[i].x, 'Point y:', polylines[i].y)
        self.construct_equality_constraints()

    def construct_equality_constraints(self):
        for i in range(1, len(self.polylines)):
            p1 = self.polylines[i-1]
            p2 = self.polylines[i]
            # equalcd
            dx = p1.x - p2.x
            dy = p1.y - p2.y
            c1 = dy
            c2 = -dx
            c3 = p1.y*dx - p1.x*dy
            valid_p = self.valid_points[i]
            # c1 x + c2 y + c3 = 0
            if c1*valid_p.x + c2*valid_p.y + c3 > 0:
                print("not valid")
                c1 = -c1
                c2 = -c2
                c3 = -c3
            # maybe noramlize?
            c_all = np.array([c1, c2, c3]) / \
                np.linalg.norm(np.array([c1, c2, c3]))

            self.inequality_constrains.append(c_all)
        print("inequality constrains:")
        for i in range(0, len(self.inequality_constrains)):
            print(self.inequality_constrains[i])

    def is_inside(self, point: Point, index):
        # point = np.array([x, y])
        c = self.inequality_constrains[index]
        if c[0]*point.x + c[1]*point.y + c[2] > 0:
            return False
        return True

    def get_inequality_cost(self, point: Point, index):
        c = self.inequality_constrains[index]
        return c[0]*point.x + c[1]*point.y + c[2]

    def get_inequality_cost_derivatives(self, index):
        # for linear inequality constrains, the derivative is the normal vector
        # tobe checked the direction
        c = self.inequality_constrains[index]
        derivate = np.array([-c[0], -c[1]])
        return derivate

    def get_near_constrains(self, pts: list[Point]):
        # return cooresponding constrain index
        # initialize all to -1
        self.pts_index_map = [0 for x in range(0, len(pts))]
        check_index = 0
        for item in pts:
            if item.x < self.polylines[0].x:
                self.pts_index_map[check_index] = 0
                check_index += 1
                continue
            if item.x > self.polylines[-1].x:
                self.pts_index_map[check_index] = len(self.polylines)-2
                check_index += 1
                continue

            for i in range(1, len(self.polylines)):
                if item.x >= self.polylines[i-1].x and item.x <= self.polylines[i].x:
                    self.pts_index_map[check_index] = (i-1)
                    break
            check_index += 1

        return self.pts_index_map

    def violate_constrains_points(self, pts: list[Point]):
        # return True if violate

        self.get_near_constrains(pts)

        if self.pts_index_map is None:
            print("somethin wrong")

        # key is the index of the point, value is the index of the constrain
        violate_dict = {}

        for i in range(0, len(pts)):
            if self.is_inside(pts[i], self.pts_index_map[i]) != True:
                violate_dict[i] = self.pts_index_map[i]

        return violate_dict

    def get_constrains(self, idxs: list[int]):
        rst = []
        for i in idxs:
            if i >= len(self.inequality_constrains):
                print("index out of range")
                return None
            rst.append(self.inequality_constrains[i])
        return rst
