from proper import *


def pinhole(wavelength, gridsize, PASSVALUE={"hole_diameter": 10e-6}):
    beam_ratio = 0.005
    focal_length = 0.5  # 焦距（镜头到检测平面的距离）0.5m
    distance = 0.1  # 光源到小孔的距离，0.1m
    wfo = prop_begin(15e-6, wavelength, gridsize, beam_ratio)
    prop_propagate(wfo, distance)

    # 圆形
    # prop_circular_aperture(wfo, PASSVALUE['hole_diameter']/2)

    # 矩形
    width = 10e-6
    height = 20e-6
    prop_rectangular_aperture(wfo, width, height)

    prop_define_entrance(wfo)
    prop_propagate(wfo, focal_length)
    (wfo, sampling) = prop_end(wfo)
    return (wfo, sampling)
