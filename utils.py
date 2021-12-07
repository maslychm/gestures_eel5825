# This file has a collection of functions for such things as
# feature extraction, processing, pre-processing, measuring
# and other things related to gesture recognition that I end up
# using in this work

import numpy as np
from numpy.linalg import norm as l2norm

from random import uniform


def path_length(pts):
    """Path traveled by the points of a gesture: sum of Euclidean
    distances between each consecutive pair of points"""
    ret = 0
    for idx in range(1, len(pts)):
        ret += l2norm(pts[idx] - pts[idx - 1])

    return ret


def resample(pts, n, variance=0.0):
    """Resample a trajectory in pts into n points"""

    # create random intervals
    scale = (12 * variance) ** .5
    intervals = [1.0 + uniform(0, 1) * scale for ii in range(n - 1)]
    total = sum(intervals)
    intervals = [val / total for val in intervals]

    # Setup place to store resampled points, and
    # store first point. jj indexes the return matrix
    ret = np.empty((n, len(pts[0])))
    ret[0] = pts[0]
    path_distance = path_length(pts)
    jj = 1

    # now do resampling
    accumulated_distance = 0.0
    interval = path_distance * intervals[jj - 1]

    for ii in range(1, len(pts)):

        distance = l2norm(pts[ii] - pts[ii - 1])

        if accumulated_distance + distance < interval:
            accumulated_distance += distance
            continue

        previous = pts[ii - 1]
        while accumulated_distance + distance >= interval:

            # Now we need to interpolate between the last point
            # and the current point.
            remaining = interval - accumulated_distance
            t = remaining / distance

            # Handle any precision errors. Note that the distance can
            # be zero if two samples are sufficiently close together,
            # which can result in nan.
            t = min(max(t, 0.0), 1.0)
            if not np.isfinite(t):
                t = 0.5

            ret[jj] = (1.0 - t) * previous + t * pts[ii]

            # Reduce the distances based on how much path we
            # just consumed.
            distance = distance - remaining
            accumulated_distance = 0.0
            previous = ret[jj]
            jj += 1

            # Exit early so we don't go past end
            # of the intervals array.
            if jj == n:
                break

            # select next interval
            interval = path_distance * intervals[jj - 1]

        accumulated_distance = distance

    if jj < n:
        ret[n - 1] = pts[ii - 1]
        jj += 1

    assert jj == n
    return ret


def normalize_vector(vec: list):
    return vec / l2norm(vec)


def get_cma_r(fs, fc):
    """"""
    m = 3.0 / 2.0 * (fs / (2.0 * np.pi * fc)) ** 2.0
    # print("m=", m)
    return int(np.ceil(m))


def rcma(pts, w, r=1):
    n = len(pts)
    for _ in range(r):
        ret = []
        for ii in range(n):
            tot = 0.0
            cnt = 0.0
            for jj in range(-w, w + 1):
                if ii + jj < 0:
                    continue
                if ii + jj >= n:
                    continue
                tot += pts[ii + jj]
                cnt += 1.0
            ret += [tot / cnt]
        pts = np.array(ret)
    return pts


def distance(pt1, pt2):
    return l2norm(pt1 - pt2)


def dot(v1, v2):
    return np.dot(v1, v2)


def vectorize(pts, normalize: bool = True):
    vecs = []
    for idx in range(1, len(pts)):
        vec = pts[idx] - pts[idx - 1]
        vec = normalize_vector(vec) if normalize else vec
        vecs.append(vec)

    return np.array(vecs)


def avg_pairwise_ip(vecs1, vecs2, square=False):
    score = 0.0
    for idx in range(len(vecs1)):
        dot = np.dot(vecs1[idx], vecs2[idx])
        if square:
            dot = dot * np.abs(dot)
        score += dot
    return score / len(vecs1)
    # return vecs1.sum(axis=None) / len(vecs1) # just the mean of inner products
    # return np.tensordot(vecs1, vecs2) # same thing but a different one-liner


def avg_pairwise_ed(t1, t2, square=False):
    score = 0.0
    for idx in range(len(t1)):
        dist = np.linalg.norm(t1[idx] - t2[idx])
        if square:
            dist = dist * np.abs(dist)
        score += dist
    return score / len(t1)


def scale_to_unit_square(pts):
    minimum = pts.min(axis=0)
    pts = pts - minimum
    maximum = pts.max(axis=0)
    return pts / maximum


def scale_translate_preserve_aspect(trajectory):
    trajectory -= trajectory.min(axis=0)
    return trajectory / np.max(trajectory)


def centroid_to_origin(trajectory):
    centroid = np.mean(trajectory, axis=0)
    pts = trajectory - centroid
    return pts


def indicative_angle(trajectory):
    """Get the angle between the centroid (point mean) and starting pt"""
    centroid = np.mean(trajectory, axis=0)
    return np.arctan2(centroid[1] - trajectory[0][1], centroid[0] - trajectory[0][0])


def rotate_by_angle(p, origin=(0, 0), angle=0):
    """Rotate trajectory around origin by an angle"""
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)


def turn_using_protractor(trajectory):
    angle = indicative_angle(trajectory)
    trajectory = rotate_by_angle(trajectory, angle=angle)
    return trajectory


def squeeze_aspect_rotate(trajectory):
    trajectory = turn_using_protractor(trajectory)
    trajectory = scale_translate_preserve_aspect(trajectory)
    return trajectory
