
from sionna.rt import Scene,load_scene, Transmitter, Receiver, PlanarArray, RadioMapSolver, PathSolver, transform_mesh, RadioMap, SceneObject
from sionna.rt.scene_utils import extend_scene_with_mesh

from matplotlib import pyplot as plt
import mitsuba as mi
import pandas as pd
import numpy as np
import drjit as dr



def init(dir_path, param_path):
    # load scene
    scene = load_scene(f"{dir_path}/scene.xml")

    # Configure the transmit array
    scene.tx_array = PlanarArray(num_rows=1,
                                 num_cols=1,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="iso",
                                 polarization="V")

    # Configure antenna array for all receivers
    scene.rx_array = PlanarArray(num_rows=1,
                                 num_cols=1,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="iso",
                                 polarization="V")

    # load transmitter
    param_df = pd.read_csv(param_path)
    tx_pos = param_df[['x', 'y', 'antenna_height']].values
    tx_rs_power = param_df[['rs_power(dBm)']].values.squeeze()
    scene.frequency = 0.7 * 1e9
    for i, pos in enumerate(tx_pos):
        tx = Transmitter(f"tx_{i}", position=mi.Point3f(pos), orientation=mi.Point3f(0, 0, 0),power_dbm=tx_rs_power[i])
        scene.add(tx)

    return scene

def mesh_cm(scene, rec_path):
    # Clone the terrain mesh
    measurement_surface = scene.objects["medium_dry_ground"].clone(as_mesh=True)
    # Shift the terrain upwards by 1.5 meters
    transform_mesh(measurement_surface, translation=np.array([0, 0, 1.5]))

    rm_solver = RadioMapSolver()
    rm = rm_solver(scene,
                   measurement_surface=measurement_surface,
                   samples_per_tx=10 ** 8,
                   # diffraction=True,
                   max_depth=5)

    # load receivers
    rec_df = pd.read_csv(rec_path)
    xy_points = rec_df[['x','y']].values

    face_ids = get_terrain_mesh_id(scene,
                        measurement_surface,
                        xy_points)

    return rm, face_ids

def path_cm(scene, rec_path):
    rec_df = pd.read_csv(rec_path)
    rx_pos = rec_df[['x', 'y', 'z']].values
    for i, pos in enumerate(rx_pos):
        rx = Receiver(f"rx_{i}", position=mi.Point3f(pos), orientation=mi.Point3f(0, 0, 0))
        scene.add(rx)

    rm_solver = PathSolver()
    paths = rm_solver(scene=scene,
                   samples_per_src=10 ** 6,
                   max_num_paths_per_src=10 ** 6,
                   max_depth=5,
                   los=True,
                   specular_reflection=True,
                   diffuse_reflection=False,
                   diffraction=False,
                   refraction=False,
                   synthetic_array=False,
                   edge_diffraction=False,
                   seed=41)
    return paths


def get_cm_db(radio_map:RadioMap,
                   metric:str,
                   tx:int | None = None) -> np.ndarray:
    coverage_map = radio_map.transmitter_radio_map(metric)
    coverage_map = coverage_map.numpy()
    valid = np.logical_and(coverage_map > 0., np.isfinite(coverage_map))
    coverage_map = coverage_map.copy()
    coverage_map[valid] = 10. * np.log10(coverage_map[valid])

    return coverage_map

def get_path_db(paths):
    # [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
    a_real, a_imag = paths.a
    a = a_real.numpy() + 1j * a_imag.numpy()

    # Transmit precoding
    # Assume default precoding
    # [num_rx, num_rx_ant, num_tx, num_paths]
    # a /= np.sqrt(a.shape[3])
    a = np.sum(a, axis=3)

    # Sum energy of paths
    a = np.square(np.abs(a))
    # [num_rx, num_tx]
    a = np.sum(a, axis=(1, 3))

    # Swap dims
    # [num_tx, num_rx]
    a = a.T

    # if not is_mesh:
    #     # Reshape to coverage map
    #     n = int(np.sqrt(a.shape[1]))
    #     shape = [a.shape[0], n, n]
    #     a = np.reshape(a, shape)

    a = np.array(a)
    db_a = np.where(a!=0, 10 * np.log10(a), 0)

    return db_a


def get_terrain_mesh_id(scene:Scene,
                        measurement_surface:SceneObject,
                        xy_points,
                        ray_origins_height:float | None = 1000.):

    xy_points = np.array(xy_points, dtype=np.float32)
    if xy_points.shape[1]==2:
        xy_points = xy_points.T

    xy_points_dr = mi.Point2f(xy_points)
    ray_origins = mi.Point3f(
        xy_points_dr.x,
        xy_points_dr.y,
        dr.full(mi.Float,ray_origins_height,xy_points.shape[1])
    )
    ray_directions = mi.Vector3f(0.,0.,-1.)

    rays = mi.Ray3f(ray_origins, ray_directions)
    if isinstance(measurement_surface, SceneObject):
        measurement_surface = measurement_surface.mi_mesh
    modified_scene = extend_scene_with_mesh(scene.mi_scene, measurement_surface)
    si_scene = modified_scene.ray_intersect(rays)
    si_valid = si_scene.is_valid()
    face_ids = si_scene.prim_index
    face_ids_np = np.array(face_ids, dtype=np.int32)
    return face_ids_np


def tow_compare(dict1, dict2):
    value1 = dict1['value']
    value2 = dict2['value']

    # 尽量保证dict1是实测
    # mask = value2 != 0
    # value1 = value1[mask]
    # value2 = value2[mask]

    errors = value1 - value2

    bound = 12
    mask = (errors < bound) & (errors > -bound)
    value1 = value1[mask]
    value2 = value2[mask]

    print(f"current points: {value1.shape[0]}")
    true_ratio = np.mean(mask)
    print(f"True的比例: {true_ratio:.2%}")  # 格式化为百分比

    errors = value1 - value2

    mean = np.mean(errors)
    std = np.std(errors)
    mae = np.mean(abs(errors))
    
    print(f"\n***** {dict1['name']} and {dict2['name']} Compare *****")
    print(f"Mean Error: {mean:.4f}")
    print(f"Standard Deviation: {std:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(value1, label=dict1['name'], marker='o', markersize=2, linewidth=1)
    plt.plot(value2, label=dict2['name'], marker='s', markersize=2, linewidth=1)

    plt.xlabel('Data Point Index')
    plt.ylabel('Values')
    plt.title(f"{dict1['name']} and {dict2['name']} Compare - Line Chart")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_compare(mesh_rm_db, rec_path, bt_path, cell_id):

    rec_df = pd.read_csv(rec_path)
    measure = rec_df[['cell_id','rsrp(dbm)']].values
    bt_df = pd.read_csv(bt_path, sep='\s+')
    bt_values = bt_df[['RSRP']].values


    mask = measure[:, 0] == cell_id  # 第一列是cell_id
    measure = measure[mask][:,1]
    sionna_values = mesh_rm_db[mask]
    bt_values = bt_values[mask].squeeze()

    print(f"original points: {sionna_values.shape[0]}")

    tow_compare({'name':"measure",'value': measure},{'name':"sionna",'value': sionna_values})
    # tow_compare({'name':"measure",'value': measure},{'name':"BT",'value': bt_values})


def plot_sionna_cmpare(mesh_rm_db,path_rm_db,rec_path,cell_id):
    rec_df = pd.read_csv(rec_path)
    measure = rec_df[['cell_id','rsrp(dbm)']].values
    mask = measure[:, 0] == cell_id
    mesh_rm_db = mesh_rm_db[mask]
    path_rm_db = path_rm_db[cell_id][mask]

    mask1 = mesh_rm_db != 0
    mask2 = path_rm_db != 0
    common_mask = mask1 & mask2

    mesh_rm_db = mesh_rm_db[common_mask]
    path_rm_db = path_rm_db[common_mask]

    tow_compare({'name':"sionna_cm",'value': mesh_rm_db},{'name':"sionna_path",'value': path_rm_db})


if __name__ == '__main__':
    # data dir
    dir_path = "./test_datasets/2604945/0_1"
    rec_path = f"{dir_path}/test_outdoor_with_z.csv"
    param_path = f"{dir_path}/eng_param.csv"
    bt_path = "D:\\code\\RaysimIndoor\\cmake-build\RaySimIndoor\\result\\power_resulzhegt_site_2604945_0.dat"

    scene = init(dir_path, param_path)

    mesh_rm , face_ids = mesh_cm(scene,rec_path)
    mesh_rm_db = get_cm_db(mesh_rm,"path_gain")[face_ids]
    # mesh_rm_db = get_cm_db(mesh_rm,"rss")[face_ids]
    plot_compare(mesh_rm_db,rec_path,bt_path,0)

    # path_rm = path_cm(scene,rec_path)
    # path_rm_db = get_path_db(path_rm)
    # plot_sionna_cmpare(mesh_rm_db,path_rm_db,rec_path,0)












