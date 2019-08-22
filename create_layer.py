#!python3
# encoding=utf8
import sys

"""
根据离散点数据绘制等值线,python3.7环境
"""

import conda,os
conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib
from mpl_toolkits.basemap import Basemap
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json,time,glob,datetime,ntpath,requests

import shapefile
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from scipy.interpolate import griddata

# shpfile边界
def getlatlonlimixt(shpfile):
    myshp = open('%s.shp' % shpfile, "rb")
    mydbf = open('%s.dbf' % shpfile, "rb")
    sjz = shapefile.Reader(shp=myshp, dbf=mydbf)
    lat_min = 90
    lat_max = -90
    lon_min = 180
    lon_max = -180
    for shape_rec in sjz.shapeRecords():
        pts = shape_rec.shape.points
        prt = list(shape_rec.shape.parts) + [len(pts)]
        for i in range(len(prt) - 1):
            for j in range(prt[i], prt[i+1]):
                lon_min = min(lon_min, pts[j][0])
                lon_max = max(lon_max, pts[j][0])
                lat_min = min(lat_min, pts[j][1])
                lat_max = max(lat_max, pts[j][1])

    return (lat_min, lat_max, lon_min, lon_max)

# 切割边界
def basemask(cs, ax, map, shpfile):
    myshp = open('%s.shp' % shpfile, "rb")
    mydbf = open('%s.dbf' % shpfile, "rb")
    sjz = shapefile.Reader(shp=myshp, dbf=mydbf)
    vertices = []
    codes = []
    lat_min = 90
    lat_max = -90
    lon_min = 180
    lon_max = -180
    for shape_rec in sjz.shapeRecords():
        pts = shape_rec.shape.points
        prt = list(shape_rec.shape.parts) + [len(pts)]
        for i in range(len(prt) - 1):
            for j in range(prt[i], prt[i+1]):
                vertices.append(map(pts[j][0], pts[j][1]))
                lon_min = min(lon_min, pts[j][0])
                lon_max = max(lon_max, pts[j][0])
                lat_min = min(lat_min, pts[j][1])
                lat_max = max(lat_max, pts[j][1])
            codes += [Path.MOVETO]
            codes += [Path.LINETO] * (prt[i+1] - prt[i] -2)
            codes += [Path.CLOSEPOLY]
        clip = Path(vertices, codes)
        clip = PathPatch(clip, transform=ax.transData) 

    for contour in cs.collections:
        contour.set_clip_path(clip)

# 补充经纬度范围四个顶点的值，防止插值不完整
def makeRectData_XYZ(lat_min, lat_max, lon_min, lon_max, x, y, z):
    # 把距离最近的点的值赋给顶点
    rect_latlons = [
        [lat_max, lon_min], # 左上角
        [lat_min, lon_min], # 左下角
        [lat_min, lon_max], # 右下角
        [lat_max, lon_max], # 右上角
    ]
    for lat_lon in rect_latlons:
        the_lat = lat_lon[0]
        the_lon = lat_lon[1]
        min_dists = sys.float_info.max      # 最大float值
        the_value = z[0]
        for i in range(len(x)):
            tmp_lat = y[i]
            tmp_lon = x[i]
            tmp_dists = np.sqrt((tmp_lon-the_lon)**2+(tmp_lat-the_lat)**2)
            if tmp_dists < min_dists:
                min_dists = tmp_dists
                the_value = z[i]
        x.append(the_lon)
        y.append(the_lat)
        z.append(the_value)

# 配置等值线绘制参数
def getcsconfig(hex_colors):
    rgb_list = []
    lvs = []
    for hex in hex_colors:
        h = hex.split(',')[-1].lstrip('#')
        rgb = tuple(int(h[i:i+2], 16) for i in (0, 2 ,4))
        # print hex, rgb
        rgb_list.append(rgb)
        if len(lvs) == 0:
            lvs.append(float(hex.split(',')[0]))
        lvs.append(float(hex.split(',')[1]))
    rgb_colors=np.array(rgb_list)/255.0

    val = np.array(lvs)

    cmap = mpl.colors.ListedColormap(rgb_colors, "")
    norm = mpl.colors.BoundaryNorm(val, len(val)-1)

    return (lvs, cmap, norm)

def main(hex_colors):
    with open('test_data.json', 'r') as f:
        data = json.loads(f.read())

    # 绘图的经纬度范围
    lat_min, lat_max, lon_min, lon_max = getlatlonlimixt('test_Project/test_Project')
    # 插值时设置的格点数，需要依据实际情况动态调整
    PLOT_Interval = 300

    x = data['dx']
    y = data['dy']
    z = data['dz']

    # 补充经纬度范围四个顶点的值，防止插值不完整
    makeRectData_XYZ(lat_min, lat_max, lon_min, lon_max, x, y, z)

    # fig = plt.figure()
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)

    m = Basemap(projection='merc',llcrnrlon=lon_min, urcrnrlon=lon_max,llcrnrlat=lat_min,urcrnrlat=lat_max, resolution='c')

    # 生成经纬度的栅格数据
    xi = np.linspace(lon_min, lon_max, PLOT_Interval)
    yi = np.linspace(lat_min, lat_max, PLOT_Interval)
    c_xi, c_yi = np.meshgrid(xi, yi)

    # 使用scipy.interpolate.griddata函数来将离散的数据格点插值到固定网格上［xi,yi］，
    # 'linear'为线性插值，'nearest'为临近插值，'cubic'为三角插值；详细请参看https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html#scipy.interpolate.griddata
    # zi = griddata((x, y), z, (c_xi, c_yi), method='linear')
    zi = griddata((np.array(x) ,np.array(y)), np.array(z), (c_xi, c_yi), method='linear')

    # 调整格点投影坐标,这是basemap一个值得注意的细节，因为投影格式的区别，需要将经纬度的数值［xi,yi］投影到实际的制图坐标
    x1, y1 = m(xi, yi)
    # 网格化经纬度网格：调用numpy的meshgrid函数来生成后面需要的网格化的数据格点xx,yy
    xx, yy = np.meshgrid(x1, y1)

    # 配置等值线绘制参数
    lvs, cmap ,norm = getcsconfig(hex_colors)
    cs = m.contourf(xx,yy, zi, lvs, alpha=1, norm=norm, cmap=cmap)

    # 切割地图
    basemask(cs, ax, m, 'test_Project/test_Project')
    # 删除图片上的边界线
    plt.axis('off')

    plt.show();exit(0)

    # 存图片
    plt.savefig('test.png', bbox_inches='tight', pad_inches=0, transparent=True)

if __name__ == '__main__':
    hex_colors = [
        "-9999,-50,#001081",
        "-50,-40,#0321b3",
        "-40,-30,#046fe3",
        "-30,-20,#3d97ef",
        "-20,-15,#70c0fd",
        "-15,-10,#8afdf8",
        "-10,-5,#bfffdc",
        "-5,0,#eaf93a",
        "0,5,#eac32a",
        "5,15,#e38221",
        "15,25,#d7304c",
        "25,30,#e86dc7",
        "30,35,#93256c",
        "35,37,#4f1c6b",
        "37,40,#fd2006",
        "40,9999,#d02478"
    ]
    main(hex_colors)
    
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),'done~')