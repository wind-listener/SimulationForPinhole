from proper import *

def NoCamera(wavelength, gridsize,PASSVALUE={"hole_diameter": 10e-6}):
    # 定义参数
    beam_diam_fraction = 0.005
    focal_length = 50e-3  # 透镜焦距（单位：米）
    object_distance = 220e-3  # 成像距离（单位：米）
    image_distance = focal_length*object_distance/(object_distance-focal_length)
    # 创建光学系统
    # 定义输入光场
    wfo = proper.prop_begin(PASSVALUE['hole_diameter'], wavelength, gridsize,beam_diam_fraction = beam_diam_fraction)
    # 添加针孔
    prop_circular_aperture(wfo, PASSVALUE['hole_diameter']/2)
    # 传播成像距离
    prop_propagate(wfo, object_distance)

    # # 添加光圈，光圈大小是焦距/F值
    # prop_circular_aperture(wfo, focal_length/2)
    # # 添加透镜
    # prop_lens(wfo, focal_length)
    # # 传播成像距离
    # prop_propagate(wfo, image_distance)

    # 在成像距离处获得光学场景
    (wfo, sampling) = prop_end(wfo)
    return (wfo, sampling)