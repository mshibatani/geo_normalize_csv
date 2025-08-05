import osmnx as ox
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

# フォント設定を簡素化（エラー回避）
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 対象の場所（例：東京駅）
PLACE = "みなみ野毘沙門の丘緑地, 八王子市, 日本"

def get_place_polygon(place):
    """
    osmnxを使って場所からポリゴンデータを取得する関数
    
    Args:
        place: 検索する場所名
    
    Returns:
        GeoDataFrame: 場所のポリゴンデータ
    """
    try:
        print(f"🌐 場所 '{place}' のデータをosmnxから取得中...")
        gdf = ox.geocode_to_gdf(place)
        
        if gdf.empty:
            print(f"❌ 場所 '{place}' が見つかりませんでした")
            return None
        else:
            print(f"✅ 場所 '{place}' のデータを正常に取得しました")
            return gdf
    except Exception as e:
        print(f"❌ エラー: {e}")
        return None

def determine_shape_orientation(coords):
    """
    形状の向きを判定して最適な回転角度を決定する関数
    
    Args:
        coords: 座標のリスト [(lon, lat), ...]
    
    Returns:
        float: 最適な回転角度（ラジアン）
    """
    if len(coords) < 3:
        return 0.0
    
    coords_array = np.array(coords)
    
    # 凸包を計算して形状の特徴を取得
    hull = ConvexHull(coords_array)
    hull_points = coords_array[hull.vertices]
    
    # 主成分分析で形状の主軸を求める
    centroid = np.mean(hull_points, axis=0)
    centered_points = hull_points - centroid
    
    # 共分散行列を計算
    cov_matrix = np.cov(centered_points.T)
    
    # 固有値と固有ベクトルを計算
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 最大の固有値に対応する固有ベクトルが主軸
    principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
    
    # 主軸の角度を計算
    angle = np.arctan2(principal_axis[1], principal_axis[0])
    
    # 形状が横長か縦長かを判定
    aspect_ratio = eigenvalues[np.argmax(eigenvalues)] / eigenvalues[np.argmin(eigenvalues)]
    
    # 横長の場合は水平に、縦長の場合は垂直に回転
    if aspect_ratio > 1.5:  # 横長の場合
        # 主軸を水平にする角度
        optimal_angle = -angle
    else:  # 縦長の場合
        # 主軸を垂直にする角度
        optimal_angle = np.pi/2 - angle
    
    # 角度を-90度～+90度の範囲に正規化
    while optimal_angle > np.pi/2:
        optimal_angle -= np.pi
    while optimal_angle < -np.pi/2:
        optimal_angle += np.pi
    
    return optimal_angle

def rotate_coordinates(coords, angle):
    """
    座標を指定した角度で回転する関数
    
    Args:
        coords: 座標のリスト [(lon, lat), ...]
        angle: 回転角度（ラジアン）
    
    Returns:
        list: 回転後の座標 [(x, y), ...]
    """
    if len(coords) < 3:
        return coords
    
    coords_array = np.array(coords)
    
    # 重心を計算
    centroid = np.mean(coords_array, axis=0)
    
    # 重心を原点に移動
    centered_coords = coords_array - centroid
    
    # 回転行列を作成
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    
    # 回転を適用
    rotated_coords = np.dot(centered_coords, rotation_matrix.T)
    
    # 重心を戻す
    final_coords = rotated_coords + centroid
    
    return final_coords.tolist()

def rotate_coordinates_back(coords, angle, original_centroid=None):
    """
    回転後の座標を元の座標系に戻す関数
    
    Args:
        coords: 回転後の座標のリスト [(x, y), ...] または単一の座標 (x, y)
        angle: 元の回転角度（ラジアン）
        original_centroid: 元の座標系の中心点 (省略可能)
    
    Returns:
        list または tuple: 元の座標系に戻した座標
    """
    # 単一の座標の場合、リストに変換して処理
    is_single_coord = False
    if not isinstance(coords, list) and not isinstance(coords, np.ndarray):
        coords = [coords]
        is_single_coord = True
    elif isinstance(coords, list) and len(coords) > 0 and not isinstance(coords[0], list) and not isinstance(coords[0], tuple):
        coords = [coords]
        is_single_coord = True
    
    coords_array = np.array(coords)
    
    # 中心点が指定されていない場合は計算
    if original_centroid is None:
        centroid = np.mean(coords_array, axis=0)
    else:
        centroid = np.array(original_centroid)
    
    # 重心を原点に移動
    centered_coords = coords_array - centroid
    
    # 逆回転行列を作成（-angleで回転）
    inverse_rotation_matrix = np.array([
        [np.cos(-angle), -np.sin(-angle)],
        [np.sin(-angle), np.cos(-angle)]
    ])
    
    # 逆回転を適用
    rotated_back_coords = np.dot(centered_coords, inverse_rotation_matrix.T)
    
    # 重心を戻す
    final_coords = rotated_back_coords + centroid
    
    # 元の形式に戻す
    if is_single_coord:
        return tuple(final_coords[0])
    else:
        return final_coords.tolist()

def convert_rotated_to_original(rotated_corners, angle, original_coords=None):
    """
    回転後の座標を元の座標系に戻す関数
    
    Args:
        rotated_corners: 回転後の座標を含む辞書 {"方向": (x, y), ...}
        angle: 回転角度（ラジアン）
        original_coords: 元のポリゴンの座標リスト（省略可能）
    
    Returns:
        dict: 元の座標系に戻した座標を含む辞書 {"rotated": {...}, "original": {...}, "angle": 角度, "centroid": 中心点}
    """
    # 元の座標系の中心を計算
    original_centroid = None
    if original_coords:
        original_centroid = np.mean(np.array(original_coords), axis=0)
    
    # 回転後の座標を元の座標系に戻す
    original_corners = {}
    for direction, rotated_coord in rotated_corners.items():
        original_coord = rotate_coordinates_back(rotated_coord, angle, original_centroid)
        original_corners[direction] = original_coord
        print(f"🔄 {direction}: 回転後({rotated_coord[0]:.6f}, {rotated_coord[1]:.6f}) → 元の座標系({original_coord[0]:.6f}, {original_coord[1]:.6f})")
    
    return {
        "rotated": rotated_corners,
        "original": original_corners,
        "angle": angle,
        "centroid": original_centroid.tolist() if original_centroid is not None else None
    }

def find_corners_human_way(coords):
    """
    人間の形状認識に基づいて角を見つける関数
    
    Args:
        coords: 座標のリスト [(lon, lat), ...]
    
    Returns:
        dict: 各方向の角の座標
        
    注意:
        方向マーカーの配置ルールについては README_DIRECTION_MARKERS.md を参照してください。
        
        主要な配置ルール:
        1. 対角方向（NE・NW・SE・SW）: ポリゴンの外接矩形の四隅に配置
        2. 主要方向（N・E・S・W）: 
           - E: NE-SE間の線分中点
           - W: NW-SW間の線分中点
           - N: NW-NE間の線上で、中点から垂直に引いた線との交点
           - S: SW-SE間の線上で、中点から垂直に引いた線との交点
        3. 重複防止: 特にN/S、E/Wが重複しないように垂直ベクトルの方向を明確に分ける
    """
    if len(coords) < 3:
        return None
    
    # 形状の向きを判定（デバッグ用）
    optimal_angle = determine_shape_orientation(coords)
    print(f"🔄 形状の最適回転角度: {np.degrees(optimal_angle):.1f}度")
    
    # 元の座標系で直接角を決定（回転を使わない方法）
    coords_array = np.array(coords)
    lons = coords_array[:, 0]
    lats = coords_array[:, 1]
    
    # 全体の中心を計算
    center_lon = np.mean(lons)
    center_lat = np.mean(lats)
    
    # 各方向の角を見つける（元の座標系で）
    corners = {}
    
    # まず4つの角（NE, NW, SE, SW）を計算してから、E、W、N、Sを中点として計算
    
    # 北東側の角（北東象限で最も外側）
    ne_indices = np.where((lats > center_lat) & (lons > center_lon))[0]
    if len(ne_indices) > 0:
        ne_distances = (lats[ne_indices] - center_lat) + (lons[ne_indices] - center_lon)
        ne_idx = ne_indices[np.argmax(ne_distances)]
        corners["北東"] = (lons[ne_idx], lats[ne_idx])
    
    # 北西側の角（北西象限で最も外側）
    nw_indices = np.where((lats > center_lat) & (lons < center_lon))[0]
    if len(nw_indices) > 0:
        nw_distances = (lats[nw_indices] - center_lat) + (center_lon - lons[nw_indices])
        nw_idx = nw_indices[np.argmax(nw_distances)]
        corners["北西"] = (lons[nw_idx], lats[nw_idx])
    
    # 南東側の角（南東象限で最も外側）
    se_indices = np.where((lats < center_lat) & (lons > center_lon))[0]
    if len(se_indices) > 0:
        se_distances = (center_lat - lats[se_indices]) + (lons[se_indices] - center_lon)
        se_idx = se_indices[np.argmax(se_distances)]
        corners["南東"] = (lons[se_idx], lats[se_idx])
    
    # 南西側の角（南西象限で最も外側）
    sw_indices = np.where((lats < center_lat) & (lons < center_lon))[0]
    if len(sw_indices) > 0:
        sw_distances = (center_lat - lats[sw_indices]) + (center_lon - lons[sw_indices])
        sw_idx = sw_indices[np.argmax(sw_distances)]
        corners["南西"] = (lons[sw_idx], lats[sw_idx])
    
    # 正しいE、W、N、Sの計算（ポリゴンの線上に垂直投影）
    if "北東" in corners and "南東" in corners:
        # E: NE-SE間の線上で、中点から垂直に引いた線との交点
        ne_corner = corners["北東"]
        se_corner = corners["南東"]
        center_point = ((ne_corner[0] + se_corner[0]) / 2, (ne_corner[1] + se_corner[1]) / 2)
        projected_point = find_perpendicular_projection_on_edge(ne_corner, se_corner, center_point, coords, direction_hint="east")
        corners["東"] = projected_point if projected_point else center_point

    if "北西" in corners and "南西" in corners:
        # W: NW-SW間の線上で、中点から垂直に引いた線との交点  
        nw_corner = corners["北西"]
        sw_corner = corners["南西"]
        center_point = ((nw_corner[0] + sw_corner[0]) / 2, (nw_corner[1] + sw_corner[1]) / 2)
        projected_point = find_perpendicular_projection_on_edge(nw_corner, sw_corner, center_point, coords, direction_hint="west")
        corners["西"] = projected_point if projected_point else center_point

    if "北東" in corners and "北西" in corners:
        # N: NW-NE間の線上で、中点から垂直に引いた線との交点
        nw_corner = corners["北西"]
        ne_corner = corners["北東"]
        center_point = ((nw_corner[0] + ne_corner[0]) / 2, (nw_corner[1] + ne_corner[1]) / 2)
        projected_point = find_perpendicular_projection_on_edge(nw_corner, ne_corner, center_point, coords, direction_hint="north")
        corners["北"] = projected_point if projected_point else center_point

    if "南東" in corners and "南西" in corners:
        # S: SW-SE間の線上で、中点から垂直に引いた線との交点
        # 重要: 仕様書に従い、Sは必ずSW-SE間のポリゴン線上に配置すること
        # EとSが重複しないよう、SE-NE間ではなくSW-SE間を使用する
        sw_corner = corners["南西"]
        se_corner = corners["南東"]
        
        # ポリゴン上のSW-SE間の実際の辺を取得
        sw_se_edge = find_polygon_edge_between_points(coords, sw_corner, se_corner)
        
        if sw_se_edge:
            # 辺の中点を計算
            edge_midpoint_idx = len(sw_se_edge) // 2
            if len(sw_se_edge) % 2 == 0:  # 偶数個の点がある場合
                p1 = sw_se_edge[edge_midpoint_idx - 1]
                p2 = sw_se_edge[edge_midpoint_idx]
                center_point = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            else:  # 奇数個の点がある場合
                center_point = sw_se_edge[edge_midpoint_idx]
            
            print(f"🔍 南側: ポリゴン辺上の中点=({center_point[0]:.6f}, {center_point[1]:.6f})")
            corners["南"] = center_point
        else:
            # 辺が見つからない場合は従来の方法でフォールバック
            center_point = ((sw_corner[0] + se_corner[0]) / 2, (sw_corner[1] + se_corner[1]) / 2)
            projected_point = find_perpendicular_projection_on_edge(sw_corner, se_corner, center_point, coords, direction_hint="south")
            corners["南"] = projected_point if projected_point else center_point
    
    return corners

def find_edge_midpoint_between_corners(coords, corner1, corner2):
    """
    2つの角の間の辺の中点座標を見つける関数
    
    Args:
        coords: 座標のリスト [(lon, lat), ...]
        corner1: 1つ目の角の座標 (lon, lat)
        corner2: 2つ目の角の座標 (lon, lat)
    
    Returns:
        (lon, lat): 辺の中点座標
    """
    if len(coords) < 3:
        return None
    
    # 2つの角の中点を計算
    midpoint = ((corner1[0] + corner2[0]) / 2, (corner1[1] + corner2[1]) / 2)
    
    # ポリゴンの線上の最も近い点を見つける
    return find_closest_point_on_polygon(coords, midpoint)

def find_closest_point_on_polygon(coords, target_point):
    """
    指定された点に最も近いポリゴンの線上の点を見つける関数（簡易版：頂点のみ）
    
    Args:
        coords: 座標のリスト [(lon, lat), ...]
        target_point: 目標点 (lon, lat)
    
    Returns:
        (lon, lat): ポリゴンの線上の最も近い点
    """
    if len(coords) < 3:
        return None
    
    coords_array = np.array(coords)
    lons = coords_array[:, 0]
    lats = coords_array[:, 1]
    
    # 各座標点までの距離を計算
    distances = np.sqrt((lons - target_point[0])**2 + (lats - target_point[1])**2)
    
    # 最も近い点のインデックスを取得
    closest_idx = np.argmin(distances)
    
    return (lons[closest_idx], lats[closest_idx])

def find_line_segment_midpoint(corner1, corner2):
    """
    2つの角を結ぶ線分の中点を計算
    """
    return ((corner1[0] + corner2[0]) / 2, (corner1[1] + corner2[1]) / 2)

def find_polygon_edge_between_points(coords, point1, point2):
    """
    ポリゴン上の2点間の辺を見つける関数
    
    Args:
        coords: ポリゴンの座標リスト
        point1: 1つ目の点 (lon, lat)
        point2: 2つ目の点 (lon, lat)
        
    Returns:
        list: ポリゴン上のpoint1とpoint2の間の辺を構成する点のリスト
    """
    # ポリゴンの頂点インデックスを見つける
    point1_idx = -1
    point2_idx = -1
    
    # 許容誤差
    epsilon = 1e-6
    
    for i, coord in enumerate(coords):
        # point1と一致する点を探す
        if abs(coord[0] - point1[0]) < epsilon and abs(coord[1] - point1[1]) < epsilon:
            point1_idx = i
        # point2と一致する点を探す
        if abs(coord[0] - point2[0]) < epsilon and abs(coord[1] - point2[1]) < epsilon:
            point2_idx = i
    
    # 両方の点が見つからなかった場合
    if point1_idx == -1 or point2_idx == -1:
        print(f"⚠️ ポリゴン上に指定された点が見つかりませんでした: point1_idx={point1_idx}, point2_idx={point2_idx}")
        return []
    
    # 辺を構成する点のリストを作成
    edge_points = []
    
    # ポリゴンは閉じているので、最後の点と最初の点は繋がっている
    n = len(coords)
    
    # point1からpoint2への経路を探す（時計回り）
    if point1_idx < point2_idx:
        edge_points = coords[point1_idx:point2_idx+1]
    else:
        edge_points = coords[point1_idx:] + coords[:point2_idx+1]
    
    # 経路が長すぎる場合、逆方向の経路を試す
    reverse_edge = []
    if point2_idx < point1_idx:
        reverse_edge = coords[point2_idx:point1_idx+1]
    else:
        reverse_edge = coords[point2_idx:] + coords[:point1_idx+1]
    
    # より短い経路を選択
    if len(reverse_edge) < len(edge_points):
        edge_points = list(reversed(reverse_edge))
    
    print(f"🔍 ポリゴン上の辺を特定: {len(edge_points)}点で構成")
    return edge_points

def point_to_line_segment_distance(point, line_start, line_end):
    """
    点から線分への最短距離を計算する関数
    
    Args:
        point: 点の座標 (x, y)
        line_start: 線分の始点 (x, y)
        line_end: 線分の終点 (x, y)
        
    Returns:
        float: 点から線分への最短距離
    """
    # 線分のベクトル
    line_vec = (line_end[0] - line_start[0], line_end[1] - line_start[1])
    
    # 線分の長さの2乗
    line_len_sq = line_vec[0]**2 + line_vec[1]**2
    
    # 線分の長さがほぼ0の場合、始点までの距離を返す
    if line_len_sq < 1e-10:
        return np.sqrt((point[0] - line_start[0])**2 + (point[1] - line_start[1])**2)
    
    # 点から始点へのベクトル
    point_vec = (point[0] - line_start[0], point[1] - line_start[1])
    
    # 内積を計算
    t = max(0, min(1, (point_vec[0] * line_vec[0] + point_vec[1] * line_vec[1]) / line_len_sq))
    
    # 線分上の最近接点
    proj_x = line_start[0] + t * line_vec[0]
    proj_y = line_start[1] + t * line_vec[1]
    
    # 点と最近接点の距離
    return np.sqrt((point[0] - proj_x)**2 + (point[1] - proj_y)**2)

def find_perpendicular_intersection_with_polygon(point, direction_vector, coords):
    """
    点から指定方向に垂直線を引いて、ポリゴンとの交点を見つける
    
    Args:
        point: 開始点 (lon, lat)
        direction_vector: 垂直線の方向ベクトル (dx, dy)
        coords: ポリゴンの座標リスト
    
    Returns:
        (lon, lat): 交点、見つからない場合はNone
    """
    if len(coords) < 3:
        return None
    
    # 垂直線上の点: point + t * direction_vector
    best_intersection = None
    min_distance = float('inf')
    
    # ポリゴンの各辺と垂直線の交点を計算
    for i in range(len(coords)):
        p1 = coords[i]
        p2 = coords[(i + 1) % len(coords)]
        
        # 線分p1-p2と垂直線の交点を計算
        intersection = line_intersection(point, direction_vector, p1, p2)
        if intersection:
            # 交点までの距離を計算
            distance = np.sqrt((intersection[0] - point[0])**2 + (intersection[1] - point[1])**2)
            if distance < min_distance:
                min_distance = distance
                best_intersection = intersection
    
    return best_intersection

def line_intersection(point, direction_vector, line_p1, line_p2):
    """
    点から方向ベクトルの直線と、2点を結ぶ線分の交点を計算
    """
    # 垂直線: point + t * direction_vector
    # 線分: line_p1 + s * (line_p2 - line_p1), 0 <= s <= 1
    
    dx1, dy1 = direction_vector
    dx2 = line_p2[0] - line_p1[0]
    dy2 = line_p2[1] - line_p1[1]
    
    # 平行線チェック
    denominator = dx1 * dy2 - dy1 * dx2
    if abs(denominator) < 1e-10:
        return None
    
    # 交点パラメータ計算
    dx3 = point[0] - line_p1[0]
    dy3 = point[1] - line_p1[1]
    
    s = (dx1 * dy3 - dy1 * dx3) / denominator
    
    # 線分上にあるかチェック
    if 0 <= s <= 1:
        # 交点座標計算
        intersection_x = line_p1[0] + s * dx2
        intersection_y = line_p1[1] + s * dy2
        return (intersection_x, intersection_y)
    
    return None

def find_perpendicular_projection_on_edge(corner1, corner2, target_point, coords, direction_hint=None):
    """
    真の垂直投影を計算する関数
    
    Args:
        corner1: 1つ目の角の座標 (lon, lat)
        corner2: 2つ目の角の座標 (lon, lat)
        target_point: 投影元の点 (lon, lat) - 使用されない（互換性のため保持）
        coords: ポリゴンの座標リスト
        direction_hint: 方向ヒント ("north", "south", "east", "west")
    
    Returns:
        (lon, lat): 投影点の座標
        
    実装詳細:
        README_DIRECTION_MARKERS.md に記載された方向マーカー配置アルゴリズムに従って実装
        
        1. 東西方向（E・W）: 単純に線分の中点を返す
        2. 南北方向（N・S）: 
           - 中点から垂直線を引いてポリゴンとの交点を計算
           - 方向に応じて垂直ベクトルを明確に分離（北と南で異なる方向）
           - Y方向の係数を大きく（0.005）、X方向の係数を小さく（0.001）設定して南北方向を強調
           - 交点が見つからない場合は方向に応じた代替アルゴリズムを使用
    """
    if len(coords) < 3:
        return None
    
    print(f"🔍 垂直投影デバッグ: 角1={corner1}, 角2={corner2}, 方向={direction_hint}")
    
    # EまたはWの場合：線分の中点を直接計算
    if direction_hint in ["east", "west"]:
        midpoint = find_line_segment_midpoint(corner1, corner2)
        print(f"🔍 東西方向：線分の中点を使用 = {midpoint}")
        return midpoint
    
    # NまたはSの場合：中点から垂直線を引いてポリゴンとの交点を計算
    elif direction_hint in ["north", "south"]:
        # 線分の中点を計算
        midpoint = find_line_segment_midpoint(corner1, corner2)
        
        # 線分に垂直な方向ベクトルを計算
        edge_vector = (corner2[0] - corner1[0], corner2[1] - corner1[1])
        
        # 方向に応じて垂直ベクトルを計算（より明確に分離）
        if direction_hint == "north":
            # 北向き垂直ベクトル（90度回転）- 強く上向き
            perpendicular_vector = (-edge_vector[1], edge_vector[0])
            # ベクトルの長さを正規化して方向を強調
            length = np.sqrt(perpendicular_vector[0]**2 + perpendicular_vector[1]**2)
            if length > 0:
                perpendicular_vector = (perpendicular_vector[0]/length * 0.001, perpendicular_vector[1]/length * 0.005)
        elif direction_hint == "south":
            # 南向き垂直ベクトル（-90度回転）- 強く下向き
            # 南側はSW-SE間のポリゴン線上に配置するため、垂直ベクトルを強調
            perpendicular_vector = (edge_vector[1], -edge_vector[0])
            # ベクトルの長さを正規化して方向を強調（Y方向を特に強調）
            length = np.sqrt(perpendicular_vector[0]**2 + perpendicular_vector[1]**2)
            if length > 0:
                # Y方向の係数を大きくして南方向への投影を強調
                perpendicular_vector = (perpendicular_vector[0]/length * 0.0005, perpendicular_vector[1]/length * 0.01)
        
        print(f"🔍 南北方向：中点={midpoint}, 垂直ベクトル={perpendicular_vector}")
        
        # 垂直線とポリゴンの交点を見つける
        intersection = find_perpendicular_intersection_with_polygon(midpoint, perpendicular_vector, coords)
        
        if intersection:
            print(f"🔍 交点発見: {intersection}")
            return intersection
        else:
            # 交点が見つからない場合は、方向に応じて別の方法を試す
            print(f"🔍 交点なし、別の方法を試行...")
            
            # ポリゴン上の点を探す（方向を考慮）
            coords_array = np.array(coords)
            
            if direction_hint == "north":
                # 北側：Y座標が大きい点を優先
                y_sorted = np.argsort(coords_array[:, 1])[::-1]  # Y座標の降順
                for idx in y_sorted[:5]:  # 上位5点を検討
                    point = coords_array[idx]
                    # X座標が近いかチェック
                    if abs(point[0] - midpoint[0]) < 0.0005:
                        print(f"🔍 北側の代替点を発見: {tuple(point)}")
                        return tuple(point)
            
            elif direction_hint == "south":
                # 南側：Y座標が小さい点を優先（南側の辺上の点を探す）
                y_sorted = np.argsort(coords_array[:, 1])  # Y座標の昇順
                
                # SW-SE間の線分に近い点を優先的に探す
                # corner1とcorner2がSW-SE間の線分の両端点
                sw_corner = corner1
                se_corner = corner2
                
                # 南側の辺（SW-SE間）に近い点を探す
                best_point = None
                min_dist = float('inf')
                
                # Y座標が小さい上位10点を検討
                for idx in y_sorted[:10]:
                    point = coords_array[idx]
                    
                    # 点から線分への距離を計算
                    dist = point_to_line_segment_distance(point, sw_corner, se_corner)
                    
                    # より線分に近い点を選択
                    if dist < min_dist:
                        min_dist = dist
                        best_point = tuple(point)
                
                if best_point:
                    print(f"🔍 南側の代替点を発見（SW-SE線分に近い点）: {best_point}")
                    return best_point
                
                # 通常のフォールバック：Y座標が小さく、X座標が中点に近い点
                for idx in y_sorted[:5]:  # 上位5点を検討
                    point = coords_array[idx]
                    # X座標が近いかチェック
                    if abs(point[0] - midpoint[0]) < 0.0005:
                        print(f"🔍 南側の代替点を発見: {tuple(point)}")
                        return tuple(point)
            
            # それでも見つからない場合は中点を使用
            print(f"🔍 代替点も見つからず、中点を使用: {midpoint}")
            return midpoint
    
    # その他の場合：中点を返す
    midpoint = find_line_segment_midpoint(corner1, corner2)
    print(f"🔍 その他：中点を使用 = {midpoint}")
    return midpoint

def calculate_direction_edge_position(gdf, direction):
    """
    GeoDataFrameから指定された方角の辺上の位置座標を計算する関数（人間の形状認識ベース）
    
    Args:
        gdf: GeoDataFrame
        direction: 方角 ("東", "西", "南", "北", "北東", "北西", "南東", "南西")
    
    Returns:
        (lon, lat): 辺上の位置座標（ポリゴンの線上）
    """
    if gdf.empty:
        return None
    
    # 座標を取得
    geometry = gdf.geometry.iloc[0]
    if hasattr(geometry, 'exterior'):
        coords = list(geometry.exterior.coords)
    else:
        return None
    
    # 形状の向きを判定して最適な回転角度を決定
    optimal_angle = determine_shape_orientation(coords)
    
    # 座標を回転
    rotated_coords = rotate_coordinates(coords, optimal_angle)
    
    # 人間の形状認識に基づいて角を見つける（回転後の座標系で）
    corners = find_corners_human_way(rotated_coords)
    if not corners:
        return None
    
    # 方角に基づいて位置を決定（ポリゴンの線上に垂直投影）
    if direction == "東":
        # 東側の辺（SE-NE間の線上）
        if "南東" in corners and "北東" in corners:
            se_corner = corners["南東"]
            ne_corner = corners["北東"]
            center_point = ((se_corner[0] + ne_corner[0]) / 2, (se_corner[1] + ne_corner[1]) / 2)
            return find_perpendicular_projection_on_edge(se_corner, ne_corner, center_point, rotated_coords, direction_hint="east")
        else:
            return corners.get("東")
    
    elif direction == "西":
        # 西側の辺（SW-NW間の線上）
        if "南西" in corners and "北西" in corners:
            sw_corner = corners["南西"]
            nw_corner = corners["北西"]
            center_point = ((sw_corner[0] + nw_corner[0]) / 2, (sw_corner[1] + nw_corner[1]) / 2)
            return find_perpendicular_projection_on_edge(sw_corner, nw_corner, center_point, rotated_coords, direction_hint="west")
        else:
            return corners.get("西")
    
    elif direction == "北":
        # 北側の辺（NW-NE間の線上）
        if "北西" in corners and "北東" in corners:
            nw_corner = corners["北西"]
            ne_corner = corners["北東"]
            center_point = ((nw_corner[0] + ne_corner[0]) / 2, (nw_corner[1] + ne_corner[1]) / 2)
            line_point = find_perpendicular_projection_on_edge(nw_corner, ne_corner, center_point, rotated_coords, direction_hint="north")
            print(f"🔍 デバッグ: 北側計算 - NW: {nw_corner}, NE: {ne_corner}, 垂直投影: {line_point}")
            return line_point
        else:
            fallback = corners.get("北")
            print(f"🔍 デバッグ: 北側計算 - フォールバック使用: {fallback}")
            return fallback
    
    elif direction == "南":
        # 南側の辺（SW-SE間の線上）
        # 重要: 仕様書に従い、Sは必ずSW-SE間のポリゴン線上に配置する
        # EとSが重複しないよう、SE-NE間ではなくSW-SE間を使用する
        if "南西" in corners and "南東" in corners:
            sw_corner = corners["南西"]
            se_corner = corners["南東"]
            
            # ポリゴン上のSW-SE間の実際の辺を取得
            sw_se_edge = find_polygon_edge_between_points(rotated_coords, sw_corner, se_corner)
            
            if sw_se_edge:
                # 辺の中点を計算
                edge_midpoint_idx = len(sw_se_edge) // 2
                if len(sw_se_edge) % 2 == 0:  # 偶数個の点がある場合
                    p1 = sw_se_edge[edge_midpoint_idx - 1]
                    p2 = sw_se_edge[edge_midpoint_idx]
                    center_point = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
                else:  # 奇数個の点がある場合
                    center_point = sw_se_edge[edge_midpoint_idx]
                
                print(f"🔍 デバッグ: 南側計算 - SW: {sw_corner}, SE: {se_corner}")
                print(f"🔍 デバッグ: 南側計算 - ポリゴン辺上の中点: {center_point}, 辺の点数: {len(sw_se_edge)}")
                return center_point
            else:
                # 辺が見つからない場合は従来の方法でフォールバック
                center_point = ((sw_corner[0] + se_corner[0]) / 2, (sw_corner[1] + se_corner[1]) / 2)
                line_point = find_perpendicular_projection_on_edge(sw_corner, se_corner, center_point, rotated_coords, direction_hint="south")
                print(f"🔍 デバッグ: 南側計算 - SW: {sw_corner}, SE: {se_corner}, 垂直投影: {line_point}")
                return line_point
        else:
            fallback = corners.get("南")
            print(f"🔍 デバッグ: 南側計算 - フォールバック使用: {fallback}")
            return fallback
    
    elif direction == "北東":
        # 北東側の角を使用
        return corners.get("北東")
    
    elif direction == "北西":
        # 北西側の角を使用
        return corners.get("北西")
    
    elif direction == "南東":
        # 南東側の角を使用
        return corners.get("南東")
    
    elif direction == "南西":
        # 南西側の角を使用
        return corners.get("南西")
    
    return None

def get_location_by_direction(place, direction):
    """
    場所名と方角を指定して、その方角の辺上の位置座標を取得する関数
    
    Args:
        place: 検索する場所名
        direction: 方角 ("東", "西", "南", "北", "北東", "北西", "南東", "南西")
    
    Returns:
        (lon, lat): 辺上の位置座標（ポリゴンの線上、回転前の元の座標系）
    """
    # osmnxでデータを取得
    gdf = get_place_polygon(place)
    
    if gdf is not None:
        # 方角別の辺上位置座標を計算
        edge_position = calculate_direction_edge_position(gdf, direction)
        
        if edge_position:
            print(f"📍 {direction}側の辺上位置座標: 経度={edge_position[0]:.6f}, 緯度={edge_position[1]:.6f}")
            return edge_position
        else:
            print(f"❌ {direction}側の辺上位置座標を計算できませんでした")
            return None
    else:
        print(f"❌ 場所 '{place}' のデータを取得できませんでした")
        return None

def visualize_with_direction_positions(place):
    """
    場所名を指定して、全方角の辺上位置座標を可視化する関数
    """
    # osmnxでデータを取得
    gdf = get_place_polygon(place)
    
    if gdf is not None:
        # 座標を取得
        geometry = gdf.geometry.iloc[0]
        if hasattr(geometry, 'exterior'):
            coords = list(geometry.exterior.coords)
        else:
            print("❌ ポリゴン情報が見つかりませんでした")
            return
        
        # 可視化
        plt.figure(figsize=(12, 10))
        
        # 元のポリゴンを描画
        lons, lats = zip(*coords)
        plt.plot(lons, lats, 'b-', linewidth=3, label='Original Polygon', alpha=0.7)
        plt.fill(lons, lats, 'blue', alpha=0.1)
        
        # 全体の中心をプロット
        center_lon = np.mean(lons)
        center_lat = np.mean(lats)
        plt.plot(center_lon, center_lat, 'ko', markersize=15, label='Center', zorder=10)
        
        # 人間の形状認識に基づいて角を見つけてプロット
        corners = find_corners_human_way(coords)
        if corners:
            corner_lons = [corner[0] for corner in corners.values()]
            corner_lats = [corner[1] for corner in corners.values()]
            plt.plot(corner_lons, corner_lats, 'rx', markersize=15, label='Corners (Human Way)', zorder=12)
        
        # 各方角の辺上位置座標を計算・プロット
        directions = ["東", "西", "南", "北", "北東", "北西", "南東", "南西"]
        direction_labels = ["E", "W", "S", "N", "NE", "NW", "SE", "SW"]
        direction_colors = {
            "東": "red", "西": "blue", "南": "green", "北": "purple",
            "北東": "orange", "北西": "brown", "南東": "pink", "南西": "cyan"
        }
        
        # ラベルの位置オフセットを定義（調整版）
        label_offsets = {
            "東": (0.0005, 0),      # 右に少しずらす
            "西": (-0.0005, 0),     # 左に少しずらす
            "南": (0.0002, -0.0005), # 下に少しずらす（右にも少し）
            "北": (0, 0.0005),      # 上に少しずらす
            "北東": (0.0003, 0.0003), # 右上にずらす
            "北西": (0.0002, 0.0003), # 左上にずらす（右にも少し）
            "南東": (0.0003, -0.0003), # 右下にずらす
            "南西": (-0.0003, -0.0003) # 左下にずらす
        }
        
        for i, direction in enumerate(directions):
            edge_position = calculate_direction_edge_position(gdf, direction)
            if edge_position:
                # NE, NW, SE, SWは角として表示
                if direction in ["北東", "北西", "南東", "南西"]:
                    plt.plot(edge_position[0], edge_position[1], 's', 
                            color=direction_colors[direction], markersize=12, 
                            markeredgecolor='white', markeredgewidth=2,
                            label=f'{direction_labels[i]} Corner', zorder=15)
                else:
                    # N, E, S, Wは中央計算として表示
                    plt.plot(edge_position[0], edge_position[1], 'o', 
                            color=direction_colors[direction], markersize=12, 
                            markeredgecolor='white', markeredgewidth=2,
                            label=f'{direction_labels[i]} Center', zorder=15)
                
                # ラベルを追加（オフセット付き）
                offset = label_offsets[direction]
                label_x = float(edge_position[0]) + offset[0]
                label_y = float(edge_position[1]) + offset[1]
                plt.text(label_x, label_y, direction_labels[i], 
                        ha='center', va='center', fontsize=10, weight='bold', 
                        color=direction_colors[direction], 
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        plt.title(f"Place: {place} Direction Positions (Human Shape Recognition)", fontsize=14)
        plt.xlabel("Longitude", fontsize=12)
        plt.ylabel("Latitude", fontsize=12)
        plt.legend(fontsize=10, loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # ファイルに保存
        output_file = f"place_{place.replace(', ', '_').replace(' ', '_')}_direction_positions_human_way.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ 可視化結果を保存しました: {output_file}")
        
        plt.show()
        
    else:
        print(f"❌ 場所 '{place}' のデータを取得できませんでした")

def visualize_rotated_positions(place):
    """
    回転後の座標系で位置を可視化する関数（デバッグ用）
    
    Args:
        place: 検索する場所名
        
    Returns:
        dict: 計算された各方向の座標（回転後の座標系）
    """
    # osmnxでデータを取得
    gdf = get_place_polygon(place)
    
    if gdf is not None:
        # 座標を取得
        geometry = gdf.geometry.iloc[0]
        if hasattr(geometry, 'exterior'):
            coords = list(geometry.exterior.coords)
        else:
            print("❌ ポリゴン情報が見つかりませんでした")
            return
        
        # 形状の向きを判定
        optimal_angle = determine_shape_orientation(coords)
        print(f"🔄 形状の最適回転角度: {np.degrees(optimal_angle):.1f}度")
        
        # 座標を回転
        rotated_coords = rotate_coordinates(coords, optimal_angle)
        rotated_array = np.array(rotated_coords)
        
        # 可視化
        plt.figure(figsize=(12, 10))
        
        # 回転後のポリゴンを描画
        lons, lats = zip(*rotated_coords)
        plt.plot(lons, lats, 'b-', linewidth=3, label='Rotated Polygon', alpha=0.7)
        plt.fill(lons, lats, 'blue', alpha=0.1)
        
        # 回転後の中心をプロット
        center_lon = np.mean(lons)
        center_lat = np.mean(lats)
        plt.plot(center_lon, center_lat, 'ko', markersize=15, label='Center', zorder=10)
        
        # 回転後の座標から角を決定
        lons_array = rotated_array[:, 0]
        lats_array = rotated_array[:, 1]
        
        # デバッグ: 回転後の座標範囲を確認
        print(f"🔍 デバッグ: 回転後の座標範囲 - X: {min(lons_array):.6f} ～ {max(lons_array):.6f}, Y: {min(lats_array):.6f} ～ {max(lats_array):.6f}")
        print(f"🔍 デバッグ: 回転後の中心座標 - ({center_lon:.6f}, {center_lat:.6f})")
        
        # 各方向の角を見つける（回転後の座標系で）
        corners = {}
        
        # まず4つの角（NE, NW, SE, SW）を計算してから、E、W、N、Sを中点として計算
        
        # 北東側の角（北東象限で最も外側）
        ne_indices = np.where((lats_array > center_lat) & (lons_array > center_lon))[0]
        if len(ne_indices) > 0:
            ne_distances = (lats_array[ne_indices] - center_lat) + (lons_array[ne_indices] - center_lon)
            ne_idx = ne_indices[np.argmax(ne_distances)]
            corners["北東"] = (lons_array[ne_idx], lats_array[ne_idx])
        
        # 北西側の角（北西象限で最も外側）
        nw_indices = np.where((lats_array > center_lat) & (lons_array < center_lon))[0]
        if len(nw_indices) > 0:
            nw_distances = (lats_array[nw_indices] - center_lat) + (center_lon - lons_array[nw_indices])
            nw_idx = nw_indices[np.argmax(nw_distances)]
            corners["北西"] = (lons_array[nw_idx], lats_array[nw_idx])
        
        # 南東側の角（南東象限で最も外側）
        se_indices = np.where((lats_array < center_lat) & (lons_array > center_lon))[0]
        if len(se_indices) > 0:
            se_distances = (center_lat - lats_array[se_indices]) + (lons_array[se_indices] - center_lon)
            se_idx = se_indices[np.argmax(se_distances)]
            corners["南東"] = (lons_array[se_idx], lats_array[se_idx])
        
        # 南西側の角（南西象限で最も外側）
        sw_indices = np.where((lats_array < center_lat) & (lons_array < center_lon))[0]
        if len(sw_indices) > 0:
            sw_distances = (center_lat - lats_array[sw_indices]) + (center_lon - lons_array[sw_indices])
            sw_idx = sw_indices[np.argmax(sw_distances)]
            corners["南西"] = (lons_array[sw_idx], lats_array[sw_idx])
        
        # デバッグ: 各角の位置を確認
        print(f"🔍 角の確認: NE=({corners.get('北東', 'なし')}), NW=({corners.get('北西', 'なし')})")
        print(f"🔍 角の確認: SE=({corners.get('南東', 'なし')}), SW=({corners.get('南西', 'なし')})")
        
        # 正しいE、W、N、Sの計算（ポリゴンの線上に垂直投影）
        print(f"🔍 東側計算開始...")
        if "北東" in corners and "南東" in corners:
            # E: NE-SE間の線上（東側の辺）
            ne_corner = corners["北東"]
            se_corner = corners["南東"]
            center_point = ((ne_corner[0] + se_corner[0]) / 2, (ne_corner[1] + se_corner[1]) / 2)
            print(f"🔍 東側: NE=({ne_corner[0]:.6f}, {ne_corner[1]:.6f}), SE=({se_corner[0]:.6f}, {se_corner[1]:.6f}), 中点=({center_point[0]:.6f}, {center_point[1]:.6f})")
            projected_point = find_perpendicular_projection_on_edge(ne_corner, se_corner, center_point, rotated_coords, "east")
            if projected_point:
                corners["東"] = projected_point
                print(f"🔍 デバッグ: 東側の垂直投影 - 座標: ({corners['東'][0]:.6f}, {corners['東'][1]:.6f})")
            else:
                corners["東"] = center_point  # フォールバック
                print(f"🔍 デバッグ: 東側の垂直投影失敗、中点使用 - 座標: ({corners['東'][0]:.6f}, {corners['東'][1]:.6f})")

        print(f"🔍 西側計算開始...")
        if "北西" in corners and "南西" in corners:
            # W: NW-SW間の線上（西側の辺）
            nw_corner = corners["北西"]
            sw_corner = corners["南西"]
            center_point = ((nw_corner[0] + sw_corner[0]) / 2, (nw_corner[1] + sw_corner[1]) / 2)
            print(f"🔍 西側: NW=({nw_corner[0]:.6f}, {nw_corner[1]:.6f}), SW=({sw_corner[0]:.6f}, {sw_corner[1]:.6f}), 中点=({center_point[0]:.6f}, {center_point[1]:.6f})")
            projected_point = find_perpendicular_projection_on_edge(nw_corner, sw_corner, center_point, rotated_coords, "west")
            if projected_point:
                corners["西"] = projected_point
                print(f"🔍 デバッグ: 西側の垂直投影 - 座標: ({corners['西'][0]:.6f}, {corners['西'][1]:.6f})")
            else:
                corners["西"] = center_point  # フォールバック
                print(f"🔍 デバッグ: 西側の垂直投影失敗、中点使用 - 座標: ({corners['西'][0]:.6f}, {corners['西'][1]:.6f})")

        print(f"🔍 北側計算開始...")
        if "北東" in corners and "北西" in corners:
            # N: NW-NE間の線上（北側の辺）
            nw_corner = corners["北西"]
            ne_corner = corners["北東"]
            center_point = ((nw_corner[0] + ne_corner[0]) / 2, (nw_corner[1] + ne_corner[1]) / 2)
            print(f"🔍 北側: NW=({nw_corner[0]:.6f}, {nw_corner[1]:.6f}), NE=({ne_corner[0]:.6f}, {ne_corner[1]:.6f}), 中点=({center_point[0]:.6f}, {center_point[1]:.6f})")
            projected_point = find_perpendicular_projection_on_edge(nw_corner, ne_corner, center_point, rotated_coords, "north")
            if projected_point:
                corners["北"] = projected_point
                print(f"🔍 デバッグ: 北側の垂直投影 - 座標: ({corners['北'][0]:.6f}, {corners['北'][1]:.6f})")
            else:
                corners["北"] = center_point  # フォールバック
                print(f"🔍 デバッグ: 北側の垂直投影失敗、中点使用 - 座標: ({corners['北'][0]:.6f}, {corners['北'][1]:.6f})")

        print(f"🔍 南側計算開始...")
        if "南東" in corners and "南西" in corners:
            # S: SW-SE間の線上（南側の辺）
            # 重要: 仕様書に従い、Sは必ずSW-SE間のポリゴン線上に配置すること
            # EとSが重複しないよう、SE-NE間ではなくSW-SE間を使用する
            sw_corner = corners["南西"]
            se_corner = corners["南東"]
            
            # ポリゴン上のSW-SE間の実際の辺を取得
            sw_se_edge = find_polygon_edge_between_points(rotated_coords, sw_corner, se_corner)
            
            if sw_se_edge:
                # 辺の中点を計算
                edge_midpoint_idx = len(sw_se_edge) // 2
                if len(sw_se_edge) % 2 == 0:  # 偶数個の点がある場合
                    p1 = sw_se_edge[edge_midpoint_idx - 1]
                    p2 = sw_se_edge[edge_midpoint_idx]
                    center_point = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
                else:  # 奇数個の点がある場合
                    center_point = sw_se_edge[edge_midpoint_idx]
                
                corners["南"] = center_point
                print(f"🔍 南側: SW=({sw_corner[0]:.6f}, {sw_corner[1]:.6f}), SE=({se_corner[0]:.6f}, {se_corner[1]:.6f})")
                print(f"🔍 南側: ポリゴン辺上の中点=({center_point[0]:.6f}, {center_point[1]:.6f}), 辺の点数={len(sw_se_edge)}")
            else:
                # 辺が見つからない場合は従来の方法でフォールバック
                center_point = ((sw_corner[0] + se_corner[0]) / 2, (sw_corner[1] + se_corner[1]) / 2)
                print(f"🔍 南側: SW=({sw_corner[0]:.6f}, {sw_corner[1]:.6f}), SE=({se_corner[0]:.6f}, {se_corner[1]:.6f}), 中点=({center_point[0]:.6f}, {center_point[1]:.6f})")
                projected_point = find_perpendicular_projection_on_edge(sw_corner, se_corner, center_point, rotated_coords, "south")
                if projected_point:
                    corners["南"] = projected_point
                    print(f"🔍 デバッグ: 南側の垂直投影 - 座標: ({corners['南'][0]:.6f}, {corners['南'][1]:.6f})")
                else:
                    corners["南"] = center_point  # フォールバック
                    print(f"🔍 デバッグ: 南側の垂直投影失敗、中点使用 - 座標: ({corners['南'][0]:.6f}, {corners['南'][1]:.6f})")
        
        # デバッグ用：SW-SE間のポリゴン辺を強調表示（赤色の太い線）
        if "南東" in corners and "南西" in corners:
            se_corner = corners["南東"]
            sw_corner = corners["南西"]
            
            # ポリゴン上のSW-SE間の実際の辺を取得
            sw_se_edge = find_polygon_edge_between_points(rotated_coords, sw_corner, se_corner)
            
            if sw_se_edge and len(sw_se_edge) > 1:
                # 辺の各セグメントを赤色の太い線で描画
                edge_lons = [p[0] for p in sw_se_edge]
                edge_lats = [p[1] for p in sw_se_edge]
                plt.plot(edge_lons, edge_lats, 
                         'r-', linewidth=8, label='SW-SE Polygon Edge (Debug)', alpha=1.0, zorder=15)
                
                # 辺の各点に小さなマーカーを追加
                plt.plot(edge_lons, edge_lats, 
                         'ro', markersize=6, markeredgecolor='black', markeredgewidth=1, 
                         alpha=0.7, zorder=16)
                
                # 辺の両端に大きなマーカーを追加
                plt.plot([sw_se_edge[0][0], sw_se_edge[-1][0]], [sw_se_edge[0][1], sw_se_edge[-1][1]], 
                         'ro', markersize=12, markeredgecolor='black', markeredgewidth=2, 
                         alpha=1.0, zorder=17)
            else:
                # 辺が見つからない場合は直線で代用
                plt.plot([se_corner[0], sw_corner[0]], [se_corner[1], sw_corner[1]], 
                         'r-', linewidth=8, label='SE-SW Line (Debug)', alpha=1.0, zorder=15)
                
                # 線分の両端に大きなマーカーを追加
                plt.plot([se_corner[0], sw_corner[0]], [se_corner[1], sw_corner[1]], 
                         'ro', markersize=12, markeredgecolor='black', markeredgewidth=2, 
                         alpha=1.0, zorder=16)

        # 角をプロット
        if corners:
            corner_lons = [corner[0] for corner in corners.values()]
            corner_lats = [corner[1] for corner in corners.values()]
            plt.plot(corner_lons, corner_lats, 'rx', markersize=15, label='Corners (Rotated)', zorder=12)
            
            # デバッグ: corners辞書の内容を確認
            print(f"🔍 デバッグ: corners辞書の内容 - {list(corners.keys())}")
            for direction, corner in corners.items():
                print(f"🔍 デバッグ: {direction} = ({corner[0]:.6f}, {corner[1]:.6f})")
        
        # 各方角の辺上位置座標を計算・プロット
        directions = ["東", "西", "南", "北", "北東", "北西", "南東", "南西"]
        direction_labels = ["E", "W", "S", "N", "NE", "NW", "SE", "SW"]
        direction_colors = {
            "東": "red", "西": "blue", "南": "green", "北": "purple",
            "北東": "orange", "北西": "brown", "南東": "pink", "南西": "cyan"
        }
        
        # ラベルの位置オフセットを定義
        label_offsets = {
            "東": (0.0001, 0),
            "西": (-0.0001, 0),
            "南": (0.0001, -0.0001),
            "北": (0, 0.0001),
            "北東": (0.0001, 0.0001),
            "北西": (0.0001, 0.0001),
            "南東": (0.0001, -0.0001),
            "南西": (-0.0001, -0.0001)
        }
        
        for i, direction in enumerate(directions):
            if direction in corners:
                corner = corners[direction]
                print(f"🎨 プロット: {direction} - 座標({corner[0]:.6f}, {corner[1]:.6f})")
                
                # NE, NW, SE, SWは角として表示
                if direction in ["北東", "北西", "南東", "南西"]:
                    plt.plot(corner[0], corner[1], 's', 
                            color=direction_colors[direction], markersize=16, 
                            markeredgecolor='black', markeredgewidth=3,
                            label=f'{direction_labels[i]} Corner', zorder=20)
                else:
                    # N, E, S, Wは中央計算として表示（大きく目立つように）
                    if direction == "南":
                        # 南側（S）は特に目立つように
                        plt.plot(corner[0], corner[1], '*', 
                                color=direction_colors[direction], markersize=25, 
                                markeredgecolor='black', markeredgewidth=3,
                                label=f'{direction_labels[i]} Center', zorder=30)
                    else:
                        # その他の方向
                        plt.plot(corner[0], corner[1], 'o', 
                                color=direction_colors[direction], markersize=18, 
                                markeredgecolor='black', markeredgewidth=3,
                                label=f'{direction_labels[i]} Center', zorder=25)
                
                # ラベルを追加（より目立つように）
                offset = label_offsets[direction]
                label_x = float(corner[0]) + offset[0]
                label_y = float(corner[1]) + offset[1]
                plt.text(label_x, label_y, direction_labels[i], 
                        ha='center', va='center', fontsize=12, weight='bold', 
                        color='black', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.9, edgecolor='black', linewidth=2),
                        zorder=30)
        
        plt.title(f"Place: {place} Rotated Coordinates (Debug View)", fontsize=14)
        plt.xlabel("Rotated Longitude", fontsize=12)
        plt.ylabel("Rotated Latitude", fontsize=12)
        plt.legend(fontsize=10, loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # 可視化範囲を自動調整
        if corners:
            corner_lons = [corner[0] for corner in corners.values()]
            corner_lats = [corner[1] for corner in corners.values()]
            
            # ポリゴンと角の座標を合わせて範囲を決定
            all_lons = list(lons) + corner_lons
            all_lats = list(lats) + corner_lats
            
            lon_min, lon_max = min(all_lons), max(all_lons)
            lat_min, lat_max = min(all_lats), max(all_lats)
            
            # マージンを追加
            lon_margin = (lon_max - lon_min) * 0.1
            lat_margin = (lat_max - lat_min) * 0.1
            
            plt.xlim(lon_min - lon_margin, lon_max + lon_margin)
            plt.ylim(lat_min - lat_margin, lat_max + lat_margin)
            
            # デバッグ: 可視化範囲を確認
            print(f"🔍 デバッグ: 可視化範囲 - X: {lon_min - lon_margin:.6f} ～ {lon_max + lon_margin:.6f}, Y: {lat_min - lat_margin:.6f} ～ {lat_max + lat_margin:.6f}")
            
            # 全ての角の座標と可視化範囲内チェック
            for direction in ["東", "西", "北", "南", "北東", "北西", "南東", "南西"]:
                if direction in corners:
                    x, y = corners[direction]
                    in_x_range = (lon_min - lon_margin) <= x <= (lon_max + lon_margin)
                    in_y_range = (lat_min - lat_margin) <= y <= (lat_max + lat_margin)
                    print(f"🔍 {direction}: ({x:.6f}, {y:.6f}) - X範囲内:{in_x_range}, Y範囲内:{in_y_range}")
                else:
                    print(f"🔍 {direction}: 座標なし")
        
        # ファイルに保存
        output_file = f"place_{place.replace(', ', '_').replace(' ', '_')}_rotated_debug.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ 回転後座標の可視化結果を保存しました: {output_file}")
        
        plt.show()
        
        # 回転後の座標のみを返す
        return corners
        
    else:
        print(f"❌ 場所 '{place}' のデータを取得できませんでした")
        return None

# メイン実行部分
if __name__ == "__main__":
    print(f"🔍 場所 '{PLACE}' の座標をosmnxから取得中...")
    print("🧠 人間の形状認識に基づくアルゴリズムで各方向の位置を決定します")
    
    # 元のポリゴン座標を取得
    gdf = get_place_polygon(PLACE)
    original_coords = None
    optimal_angle = None
    
    if gdf is not None:
        geometry = gdf.geometry.iloc[0]
        if hasattr(geometry, 'exterior'):
            original_coords = list(geometry.exterior.coords)
            # 形状の向きを判定
            optimal_angle = determine_shape_orientation(original_coords)
            print(f"🔄 形状の最適回転角度: {np.degrees(optimal_angle):.1f}度")
    
    # 回転後の座標系で可視化（デバッグ用）
    print("\n🔍 回転後の座標系で可視化（デバッグ）:")
    rotated_corners = visualize_rotated_positions(PLACE)
    
    # 回転後の座標を元の座標系に戻す
    corners_data = None
    if rotated_corners and optimal_angle is not None:
        print("\n🔄 回転後の座標を元の座標系に戻します:")
        corners_data = convert_rotated_to_original(rotated_corners, optimal_angle, original_coords)
        
        # 回転後と元の座標系の座標を表示
        if corners_data:
            rotated_corners = corners_data["rotated"]
            original_corners = corners_data["original"]
            rotation_angle = corners_data["angle"]
            
            print(f"\n🔄 回転角度: {np.degrees(rotation_angle):.2f}度")
            
            print("\n💾 回転後と元の座標系の座標を保存しました:")
            print("\n| 方向 | 回転後の座標 (x, y) | 元の座標系 (lon, lat) |")
            print("|------|-------------------|-------------------|")
            for direction in ["東", "西", "南", "北", "北東", "北西", "南東", "南西"]:
                if direction in rotated_corners and direction in original_corners:
                    rotated = rotated_corners[direction]
                    original = original_corners[direction]
                    print(f"| {direction} | ({rotated[0]:.6f}, {rotated[1]:.6f}) | ({original[0]:.6f}, {original[1]:.6f}) |")
    
    # 全方角の辺上位置座標を計算（元の座標系）
    directions = ["東", "西", "南", "北", "北東", "北西", "南東", "南西"]
    
    print("\n📍 各方向の辺上位置座標（人間の形状認識ベース）:")
    api_corners = {}
    for direction in directions:
        coords = get_location_by_direction(PLACE, direction)
        if coords:
            api_corners[direction] = coords
    
    # 元の座標系の座標と比較（検証用）
    if corners_data and api_corners:
        original_corners = corners_data["original"]
        print("\n🔍 元の座標系の座標の比較（回転逆変換 vs API取得）:")
        print("\n| 方向 | 回転逆変換 (lon, lat) | API取得 (lon, lat) | 差分 (m) |")
        print("|------|-------------------|-------------------|---------|")
        for direction in directions:
            if direction in original_corners and direction in api_corners:
                rotated_back = original_corners[direction]
                api = api_corners[direction]
                # 緯度経度の差をメートルに換算（概算）
                # 緯度1度は約111km、経度1度は緯度によって異なるが、約111km×cos(緯度)
                lat_diff_m = abs(rotated_back[1] - api[1]) * 111000
                lon_diff_m = abs(rotated_back[0] - api[0]) * 111000 * np.cos(np.radians(api[1]))
                total_diff_m = np.sqrt(lat_diff_m**2 + lon_diff_m**2)
                print(f"| {direction} | ({rotated_back[0]:.6f}, {rotated_back[1]:.6f}) | ({api[0]:.6f}, {api[1]:.6f}) | {total_diff_m:.2f} |")
    
    print("\n🎨 可視化を実行中...")
    visualize_with_direction_positions(PLACE)
    
    print("\n💡 人間の形状認識に基づくアルゴリズムで高速にデータを取得しました")