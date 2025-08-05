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

def find_corners_human_way(coords):
    """
    人間の形状認識に基づいて角を見つける関数
    
    Args:
        coords: 座標のリスト [(lon, lat), ...]
    
    Returns:
        dict: 各方向の角の座標
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
    
    # 正しいE、W、N、Sの計算（角の中点として）
    if "北東" in corners and "南東" in corners:
        corners["東"] = (
            (corners["北東"][0] + corners["南東"][0]) / 2,
            (corners["北東"][1] + corners["南東"][1]) / 2
        )

    if "北西" in corners and "南西" in corners:
        corners["西"] = (
            (corners["北西"][0] + corners["南西"][0]) / 2,
            (corners["北西"][1] + corners["南西"][1]) / 2
        )

    if "北東" in corners and "北西" in corners:
        corners["北"] = (
            (corners["北東"][0] + corners["北西"][0]) / 2,
            (corners["北東"][1] + corners["北西"][1]) / 2
        )

    if "南東" in corners and "南西" in corners:
        corners["南"] = (
            (corners["南東"][0] + corners["南西"][0]) / 2,
            (corners["南東"][1] + corners["南西"][1]) / 2
        )
    
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
    指定された点に最も近いポリゴンの線上の点を見つける関数
    
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
    
    # 人間の形状認識に基づいて角を見つける
    corners = find_corners_human_way(coords)
    if not corners:
        return None
    
    # 方角に基づいて位置を決定
    if direction == "東":
        # 東側の辺（SEとNEの中央）
        if "南東" in corners and "北東" in corners:
            se_corner = corners["南東"]
            ne_corner = corners["北東"]
            midpoint = ((se_corner[0] + ne_corner[0]) / 2, (se_corner[1] + ne_corner[1]) / 2)
            return midpoint
        else:
            return corners.get("東")
    
    elif direction == "西":
        # 西側の辺（SWとNWの中央）
        if "南西" in corners and "北西" in corners:
            sw_corner = corners["南西"]
            nw_corner = corners["北西"]
            midpoint = ((sw_corner[0] + nw_corner[0]) / 2, (sw_corner[1] + nw_corner[1]) / 2)
            return midpoint
        else:
            return corners.get("西")
    
    elif direction == "北":
        # 北側の辺（NWとNEの中央）
        if "北西" in corners and "北東" in corners:
            nw_corner = corners["北西"]
            ne_corner = corners["北東"]
            midpoint = ((nw_corner[0] + ne_corner[0]) / 2, (nw_corner[1] + ne_corner[1]) / 2)
            print(f"🔍 デバッグ: 北側計算 - NW: {nw_corner}, NE: {ne_corner}, 中点: {midpoint}")
            return midpoint
        else:
            fallback = corners.get("北")
            print(f"🔍 デバッグ: 北側計算 - フォールバック使用: {fallback}")
            return fallback
    
    elif direction == "南":
        # 南側の辺（SWとSEの中央）
        if "南西" in corners and "南東" in corners:
            sw_corner = corners["南西"]
            se_corner = corners["南東"]
            midpoint = ((sw_corner[0] + se_corner[0]) / 2, (sw_corner[1] + se_corner[1]) / 2)
            print(f"🔍 デバッグ: 南側計算 - SW: {sw_corner}, SE: {se_corner}, 中点: {midpoint}")
            return midpoint
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
        (lon, lat): 辺上の位置座標（ポリゴンの線上）
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
        
        # 正しいE、W、N、Sの計算（角の中点として）
        if "北東" in corners and "南東" in corners:
            corners["東"] = (
                (corners["北東"][0] + corners["南東"][0]) / 2,
                (corners["北東"][1] + corners["南東"][1]) / 2
            )
            print(f"🔍 デバッグ: 東側の中点 - 座標: ({corners['東'][0]:.6f}, {corners['東'][1]:.6f})")

        if "北西" in corners and "南西" in corners:
            corners["西"] = (
                (corners["北西"][0] + corners["南西"][0]) / 2,
                (corners["北西"][1] + corners["南西"][1]) / 2
            )
            print(f"🔍 デバッグ: 西側の中点 - 座標: ({corners['西'][0]:.6f}, {corners['西'][1]:.6f})")

        if "北東" in corners and "北西" in corners:
            corners["北"] = (
                (corners["北東"][0] + corners["北西"][0]) / 2,
                (corners["北東"][1] + corners["北西"][1]) / 2
            )
            print(f"🔍 デバッグ: 北側の中点 - 座標: ({corners['北'][0]:.6f}, {corners['北'][1]:.6f})")

        if "南東" in corners and "南西" in corners:
            corners["南"] = (
                (corners["南東"][0] + corners["南西"][0]) / 2,
                (corners["南東"][1] + corners["南西"][1]) / 2
            )
            print(f"🔍 デバッグ: 南側の中点 - 座標: ({corners['南'][0]:.6f}, {corners['南'][1]:.6f})")
        
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
                # NE, NW, SE, SWは角として表示
                if direction in ["北東", "北西", "南東", "南西"]:
                    plt.plot(corner[0], corner[1], 's', 
                            color=direction_colors[direction], markersize=12, 
                            markeredgecolor='white', markeredgewidth=2,
                            label=f'{direction_labels[i]} Corner', zorder=15)
                else:
                    # N, E, S, Wは中央計算として表示
                    plt.plot(corner[0], corner[1], 'o', 
                            color=direction_colors[direction], markersize=12, 
                            markeredgecolor='white', markeredgewidth=2,
                            label=f'{direction_labels[i]} Center', zorder=15)
                
                # ラベルを追加
                offset = label_offsets[direction]
                label_x = float(corner[0]) + offset[0]
                label_y = float(corner[1]) + offset[1]
                plt.text(label_x, label_y, direction_labels[i], 
                        ha='center', va='center', fontsize=10, weight='bold', 
                        color=direction_colors[direction], 
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
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
            print(f"🔍 デバッグ: EとWの座標 - E: ({corners['東'][0]:.6f}, {corners['東'][1]:.6f}), W: ({corners['西'][0]:.6f}, {corners['西'][1]:.6f})")
        
        # ファイルに保存
        output_file = f"place_{place.replace(', ', '_').replace(' ', '_')}_rotated_debug.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ 回転後座標の可視化結果を保存しました: {output_file}")
        
        plt.show()
        
    else:
        print(f"❌ 場所 '{place}' のデータを取得できませんでした")

# メイン実行部分
if __name__ == "__main__":
    print(f"🔍 場所 '{PLACE}' の座標をosmnxから取得中...")
    print("🧠 人間の形状認識に基づくアルゴリズムで各方向の位置を決定します")
    
    # 回転後の座標系で可視化（デバッグ用）
    print("\n🔍 回転後の座標系で可視化（デバッグ）:")
    visualize_rotated_positions(PLACE)
    
    # 全方角の辺上位置座標を計算
    directions = ["東", "西", "南", "北", "北東", "北西", "南東", "南西"]
    
    print("\n📍 各方向の辺上位置座標（人間の形状認識ベース）:")
    for direction in directions:
        get_location_by_direction(PLACE, direction)
    
    print("\n🎨 可視化を実行中...")
    visualize_with_direction_positions(PLACE)
    
    print("\n💡 人間の形状認識に基づくアルゴリズムで高速にデータを取得しました")