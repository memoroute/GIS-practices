import requests
from tkinter import *
from PIL import ImageTk, Image
import pickle
import os.path
import json
from tkinter import messagebox
from tkinter.ttk import Progressbar
import re

"""
交互式地图查看器，功能如下：
1. 地图操作：用户可以通过按钮或鼠标拖拽实现地图的上下左右平移、放大缩小，并实时显示鼠标所在位置的经纬度。
2. 图层控制：支持显示/隐藏多个地图图层（如城市、洲界、世界边界），并通过复选框动态更新图层可见性。
3. 要素识别：点击地图上的某一点，可识别并显示该点的地理要素信息（如所属图层和坐标）。
4. 地震数据展示：同时加载并展示地震数据图层，随主地图同步更新。
5. 要素服务管理：提供显示可用要素服务的功能，并支持下载与当前地图范围相交的矢量数据，附带进度条提示。
6. 状态保存：支持保存和加载地图状态，确保用户操作的连续性。
"""

# region 上下左右平移，放大缩小，拖拽，显示鼠标坐标，更新图层
def pan_left():
    """地图左移"""
    global map_url

    # 提取出当前显示地图的范围的经纬度值
    bbox = map_url.split('bbox=')[1].split('&')[0]
    bbox = [float(coord) for coord in bbox.split(',')]

    # 计算移动后地图显示范围的经纬度
    bbox[0] -= (bbox[2] - bbox[0]) / 500 * 10
    bbox[2] -= (bbox[2] - bbox[0]) / 500 * 10

    # 更新地图
    bbox = [str(coord) for coord in bbox]
    bbox = ','.join(bbox)
    map_url = map_url.replace('bbox=' + map_url.split('bbox=')[1].split('&')[0], 'bbox=' + bbox)

    update_map()


def pan_right():
    """地图右移"""
    global map_url
    bbox = map_url.split('bbox=')[1].split('&')[0]
    bbox = [float(coord) for coord in bbox.split(',')]
    bbox[0] += (bbox[2] - bbox[0]) / 500 * 10
    bbox[2] += (bbox[2] - bbox[0]) / 500 * 10
    bbox = [str(coord) for coord in bbox]
    bbox = ','.join(bbox)
    map_url = map_url.replace('bbox=' + map_url.split('bbox=')[1].split('&')[0], 'bbox=' + bbox)
    update_map()


def pan_up():
    """地图上移"""
    global map_url
    bbox = map_url.split('bbox=')[1].split('&')[0]
    bbox = [float(coord) for coord in bbox.split(',')]
    bbox[1] += (bbox[3] - bbox[1]) / 500 * 10
    bbox[3] += (bbox[3] - bbox[1]) / 500 * 10
    bbox = [str(coord) for coord in bbox]
    bbox = ','.join(bbox)
    map_url = map_url.replace('bbox=' + map_url.split('bbox=')[1].split('&')[0], 'bbox=' + bbox)
    update_map()


def pan_down():
    """地图下移"""
    global map_url
    bbox = map_url.split('bbox=')[1].split('&')[0]
    bbox = [float(coord) for coord in bbox.split(',')]
    bbox[1] -= (bbox[3] - bbox[1]) / 500 * 10
    bbox[3] -= (bbox[3] - bbox[1]) / 500 * 10
    bbox = [str(coord) for coord in bbox]
    bbox = ','.join(bbox)
    map_url = map_url.replace('bbox=' + map_url.split('bbox=')[1].split('&')[0], 'bbox=' + bbox)
    update_map()


def zoom_out():
    """地图缩小"""
    global map_url
    bbox = map_url.split('bbox=')[1].split('&')[0]
    bbox = [float(coord) for coord in bbox.split(',')]
    bbox[0] -= (bbox[2] - bbox[0]) / 500 * 10
    bbox[1] -= (bbox[3] - bbox[1]) / 500 * 10
    bbox[2] += (bbox[2] - bbox[0]) / 500 * 10
    bbox[3] += (bbox[3] - bbox[1]) / 500 * 10
    bbox = [str(coord) for coord in bbox]
    bbox = ','.join(bbox)
    map_url = map_url.replace('bbox=' + map_url.split('bbox=')[1].split('&')[0], 'bbox=' + bbox)
    update_map()


def zoom_in():
    """地图放大"""
    global map_url
    bbox = map_url.split('bbox=')[1].split('&')[0]
    bbox = [float(coord) for coord in bbox.split(',')]
    bbox[0] += (bbox[2] - bbox[0]) / 500 * 10
    bbox[1] += (bbox[3] - bbox[1]) / 500 * 10
    bbox[2] -= (bbox[2] - bbox[0]) / 500 * 10
    bbox[3] -= (bbox[3] - bbox[1]) / 500 * 10
    bbox = [str(coord) for coord in bbox]
    bbox = ','.join(bbox)
    map_url = map_url.replace('bbox=' + map_url.split('bbox=')[1].split('&')[0], 'bbox=' + bbox)
    update_map()


start_x = None
start_y = None


def reset_start(event):
    """重置起始点坐标"""
    global start_x, start_y
    start_x = None
    start_y = None


def pan(event):
    """地图拖拽"""
    global map_url, start_x, start_y

    if start_x and start_y:
        # 记录鼠标移动时，鼠标所在位置的x，y坐标
        end_x = event.x
        end_y = event.y

        # 解析当前显示的地图经的纬度范围
        bbox = map_url.split('bbox=')[1].split('&')[0].split(',')
        bbox = [float(coord) for coord in bbox]

        # 计算移动的经纬度
        diff_x = (end_x - start_x) * (bbox[2] - bbox[0]) / 500
        diff_y = (end_y - start_y) * (bbox[3] - bbox[1]) / 500

        # 更新地图的经纬度范围
        bbox[0] -= diff_x
        bbox[1] += diff_y
        bbox[2] -= diff_x
        bbox[3] += diff_y

        # 更新map_url
        bbox = [str(coord) for coord in bbox]
        bbox = ','.join(bbox)
        map_url = map_url.replace('bbox=' + map_url.split('bbox=')[1].split('&')[0], 'bbox=' + bbox)

        update_map()

    # 记录点下的一瞬间的点坐标值
    start_x = event.x
    start_y = event.y


def show_coordinates(event):
    """显示鼠标所在位置的坐标"""

    bbox = map_url.split('bbox=')[1].split('&')[0].split(',')
    bbox = [float(coord) for coord in bbox]

    # 计算鼠标所在位置的经纬度
    diff_x = event.x * (bbox[2] - bbox[0]) / 500
    diff_y = (500 - event.y) * (bbox[3] - bbox[1]) / 500
    longitude = bbox[0] + diff_x
    latitude = bbox[1] + diff_y

    coord_label.config(text="Longitude: {:.6f}\nLatitude: {:.6f}".format(longitude, latitude))


def update_layer_visibility():
    """更新图层"""
    global map_url, layers_visibility
    visible_layers = [index for index, var in layers_visibility.items() if var.get() == 1]
    layers_param = ",".join(visible_layers) if visible_layers else "show:"

    # 更新map_url的layers参数
    map_url = re.sub(r'layers=show:[0-9,]*', f'layers=show:{layers_param}', map_url)

    # 更新地图
    update_map()
# endregion


# region 识别并显示要素
def identify_feature(event):
    """identify函数"""
    global map_url, root
    # 点击的像素坐标
    x = event.x
    y = event.y

    # 当前地图的bbox
    bbox = map_url.split('bbox=')[1].split('&')[0]
    bbox_coords = [float(coord) for coord in bbox.split(',')]

    dx = (bbox_coords[2] - bbox_coords[0]) / 500
    dy = (bbox_coords[3] - bbox_coords[1]) / 500
    map_point_x = bbox_coords[0] + dx * x
    map_point_y = bbox_coords[3] - dy * y
    identify_params = {
        "f": "json",
        "geometry": f"{map_point_x},{map_point_y}",
        "geometryType": "esriGeometryPoint",
        "sr": "4326",
        "layers": "visible",
        "mapExtent": bbox,
        "imageDisplay": "500,500,96",
        "tolerance": "2"
    }
    identify_url = "https://sampleserver6.arcgisonline.com/arcgis/rest/services/SampleWorldCities/MapServer/identify"
    resp = requests.get(identify_url, params=identify_params)
    feature_info = resp.json()

    if feature_info['results']:
        layers = feature_info['results'][0]['layerName']
        messagebox.showinfo("Feature Info", f"Layer: {layers}\nCoordinate: {map_point_x}, {map_point_y}")
    else:
        messagebox.showinfo("Feature Info", "No features found.")
    if feature_info['results']:
        metadata_url = identify_url.rstrip("identify")
        metadata_resp = requests.get(metadata_url, params={"f": "json"})
        metadata = metadata_resp.json()
        layers = metadata['layers']
        spatial_reference = metadata['spatialReference']
        initial_extent = metadata['initialExtent']
        messagebox.showinfo("Metadata Info",
                            f"Layers: {json.dumps(layers, indent=2)}\nSpatial Reference: {spatial_reference}\nInitial Extent: {initial_extent}")
# endregion


# region 显示要素服务和下载矢量数据
def has_intersection(bbox1):
    """检测窗口中的地图范围是否和要素服务相交"""
    global map_url
    bbox = map_url.split('bbox=')[1].split('&')[0]
    bbox = [float(coord) for coord in bbox.split(',')]
    xmin = bbox[0]
    ymin = bbox[1]
    xmax = bbox[2]
    ymax = bbox[3]

    # 解析bbox坐标
    xmin1 = float(bbox1['xmin'])
    ymin1 = float(bbox1['ymin'])
    xmax1 = float(bbox1['xmax'])
    ymax1 = float(bbox1['ymax'])
    # 检查bbox是否有交集
    if (xmin1 > xmax) or (xmax1 < xmin) or (ymin1 > ymax) or (ymax1 < ymin):
        return False
    else:
        return True


def display_feature_services():
    """显示要素服务"""
    url = 'https://sampleserver6.arcgisonline.com/arcgis/rest/services'
    resp = requests.get(url, params={'f': 'json'})
    services = resp.json()['services']

    feature_services = [service for service in services if service['type'] == 'FeatureServer']

    feature_services_info = "\n".join(
        [f"Name: {service['name']}, Type: {service['type']}" for service in feature_services])

    messagebox.showinfo('Feature Services Information', f"Feature Services:\n{feature_services_info}")


def download_vector_data():
    """下载矢量数据"""

    def download_progress():
        """添加下载进度条"""
        layer_progress["value"] = 0
        progress_window.deiconify()
        progress_window.update()

        total_layers = 0
        for service in featureserver_services:
            resp = requests.get(f"{base_url}/{service['name']}/FeatureServer/layers?f=pjson")
            layers = resp.json()['layers']
            total_layers += len(layers)

        layer_count = 0

        for service in featureserver_services:
            resp = requests.get(f"{base_url}/{service['name']}/FeatureServer/layers?f=pjson")
            layers = resp.json()['layers']

            # 获取服务范围的bbox
            service_bbox = resp.json()['layers'][0]['extent']

            # 检查bbox和服务范围是否有交集
            if has_intersection(service_bbox):
                for layer in layers:
                    bbox = map_url.split('bbox=')[1].split('&')[0]
                    url = f"{base_url}/{service['name']}/FeatureServer/{layer['id']}/query"
                    params = {
                        "where": "1=1",
                        "geometry": bbox,
                        "geometryType": "esriGeometryEnvelope",
                        "inSR": "4326",
                        "spatialRel": "esriSpatialRelIntersects",
                        "outFields": "*",
                        "returnGeometry": "true",
                        "f": "geojson",
                    }
                    resp = requests.get(url, params=params)

                    with open(f"{service['name']}_{layer['name']}.json", "w") as f:
                        f.write(json.dumps(resp.json()))

                    layer_count += 1
                    layer_progress["value"] = (layer_count / total_layers) * 100
                    progress_window.update()

        progress_window.withdraw()

    global bbox
    base_url = 'https://sampleserver6.arcgisonline.com/arcgis/rest/services'
    # fetch json中的所有services的键值
    resp = requests.get(f"{base_url}?f=pjson")
    services = resp.json()['services']

    # 搜寻type为FeatureServer的服务
    featureserver_services = [service for service in services if service['type'] == 'FeatureServer']

    progress_window = Toplevel()
    progress_window.title("Download Progress")

    layer_progress = Progressbar(progress_window, orient='horizontal', length=300, mode="determinate")
    layer_progress.pack(pady=10)

    progress_label = Label(progress_window, text="Downloading vector data...")
    progress_label.pack()

    download_button = Button(progress_window, text="Cancel", command=progress_window.destroy)
    download_button.pack(pady=10)

    progress_window.after(100, download_progress)


# endregion


# region更新地图
def update_earthquake_map():
    """更新地震矢量数据地图"""
    global earthquake_map_url
    response = requests.get(earthquake_map_url)
    image_data = response.content
    with open('earthquake_map.png', 'wb') as f:
        f.write(image_data)
    img = Image.open('earthquake_map.png')
    img = img.resize((500, 500), Image.BILINEAR)
    earthquake_photo = ImageTk.PhotoImage(img)
    earthquake_map_label.config(image=earthquake_photo)
    earthquake_map_label.image = earthquake_photo


def update_map():
    """更新地图"""
    global map_url, earthquake_map_url

    # 更新主地图
    response = requests.get(map_url)
    image_data = response.content
    with open('../map.png', 'wb') as f:
        f.write(image_data)
    img = Image.open('../map.png')
    img = img.resize((500, 500), Image.BILINEAR)
    map_img = ImageTk.PhotoImage(img)
    map_label.configure(image=map_img)
    map_label.image = map_img

    # 更新全局变量 'earthquake_map_url'
    bbox = map_url.split('bbox=')[1].split('&')[0]
    earthquake_map_url = re.sub(r'bbox=[^&]*', f'bbox={bbox}', earthquake_map_url)

    # 更新地震地图
    update_earthquake_map()


# endregion

# region创建主窗口
root = Tk()
root.title('Map Viewer')
# endregion

# region地图的URL
map_url = 'https://sampleserver6.arcgisonline.com/arcgis/rest/services/SampleWorldCities/MapServer/export?bbox='
bbox = '70.32842616117323,-3.111198035376013,155.62617538039785,78.03101981162999'
map_url += bbox + '&bboxSR=4326&imageSR=4326&size=1559,1483&dpi=191.9999972950327&format=png32&transparent=true&layers=show:0,1,2&f=image'

earthquake_map_url = 'https://sampleserver6.arcgisonline.com/arcgis/rest/services/Earthquakes_Since1970/MapServer/export?bbox='
earthquake_map_url += bbox + '&bboxSR=4326&imageSR=4326&size=1559,1483&dpi=191.9999972950327&format=png32&transparent=true&layers=show:0&f=image'
# endregion

# region读取之前保存的map_url
if os.path.isfile('../map_url.pickle'):
    with open('../map_url.pickle', 'rb') as f:
        map_url = pickle.load(f)
# endregion

# region将地图显示到窗口中
response = requests.get(map_url)
image_data = response.content
with open('../map.png', 'wb') as f:
    f.write(image_data)

img = Image.open('../map.png')
img = img.resize((500, 500), Image.BILINEAR)
map_img = ImageTk.PhotoImage(img)
# endregion

# region显示地震数据的地图
response = requests.get(earthquake_map_url)
eq_image_data = response.content
with open('../earthquake_map.png', 'wb') as f:
    f.write(eq_image_data)
eq_img = Image.open('../earthquake_map.png')
eq_img = eq_img.resize((500, 500), Image.BILINEAR)
eq_map_img = ImageTk.PhotoImage(eq_img)
# endregion

# region在窗口中放置地图
# SampleWorldCities地图
map_label = Label(root, image=map_img)
map_label.pack(side=RIGHT)

# Earthquakes_Since1970地图
earthquake_map_label = Label(root, image=eq_map_img)
earthquake_map_label.pack(side=LEFT)
# endregion

# region 用于显示鼠标实时的经纬度
coord_label = Label(root)
coord_label.pack()
# endregion

# region图层的显示状态
layers_visibility = {
    "0": IntVar(value=1),  # Cities，默认显示
    "1": IntVar(value=1),  # Continent，默认显示
    "2": IntVar(value=1)   # World，默认显示
}
# endregion

# region 绑定鼠标按键功能
map_label.bind("<B1-Motion>", pan)  # 鼠标按下时，开始调用pan
map_label.bind("<ButtonRelease-1>", reset_start)  # 鼠标松开时，调用reset_start
root.bind("<Motion>", show_coordinates)  # 鼠标移动时，调用show_coordinates
map_label.bind("<Button-1>", identify_feature)
# endregion

# region 创建按钮
# 创建图层 frames 来容纳控件
control_frame = Frame(root)
control_frame.pack(side=BOTTOM, fill=X, padx=4, pady=4)

layer_controls_frame = Frame(control_frame)
layer_controls_frame.pack(side=TOP, pady=4)

pan_frame = Frame(control_frame)
pan_frame.pack(side=LEFT, padx=4)

zoom_frame = Frame(control_frame)
zoom_frame.pack(side=LEFT, padx=4)

feature_service_frame = Frame(control_frame)
feature_service_frame.pack(side=TOP, padx=4)

# 用pack布局在layer_controls_frame中排列复选框
check_city = Checkbutton(layer_controls_frame, text="Cities", variable=layers_visibility["0"], command=update_layer_visibility)
check_city.pack(side=LEFT, padx=2, anchor=W)

check_continent = Checkbutton(layer_controls_frame, text="Continent", variable=layers_visibility["1"], command=update_layer_visibility)
check_continent.pack(side=LEFT, padx=2, anchor=W)

check_world = Checkbutton(layer_controls_frame, text="World", variable=layers_visibility["2"], command=update_layer_visibility)
check_world.pack(side=LEFT, padx=2, anchor=W)


# 用grid布局在pan_frame中排列pan和zoom按钮
pan_up_btn = Button(pan_frame, text='上移', command=pan_up)
pan_up_btn.grid(row=0, column=1, sticky=W + E)

pan_left_btn = Button(pan_frame, text='左移', command=pan_left)
pan_left_btn.grid(row=1, column=0, sticky=W + E)

pan_down_btn = Button(pan_frame, text='下移', command=pan_down)
pan_down_btn.grid(row=2, column=1, sticky=W + E)

pan_right_btn = Button(pan_frame, text='右移', command=pan_right)
pan_right_btn.grid(row=1, column=2, sticky=W + E)

# 用grid布局在zoom_frame中排列zoom按钮
zoom_in_btn = Button(zoom_frame, text='放大', command=zoom_in)
zoom_in_btn.grid(row=0, column=0, padx=2)

zoom_out_btn = Button(zoom_frame, text='缩小', command=zoom_out)
zoom_out_btn.grid(row=1, column=0, padx=2)

# 创建特征服务按钮
display_btn = Button(feature_service_frame, text="显示要素", command=display_feature_services)
display_btn.grid(row=0, column=0)

download_btn = Button(feature_service_frame, text="下载要素", command=download_vector_data)
download_btn.grid(row=0, column=1)

# endregion


root.mainloop()
