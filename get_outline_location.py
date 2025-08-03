import osmnx as ox
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

# フォント設定を簡素化（エラー回避）
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 対象の場所（例：東京駅）
PLACE = "長沼公園, 八王子市, 日本"

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

def find_corners(gdf):
    """
    GeoDataFrameから各方向の角（最も外側の点）を見つける関数
    
    Args:
        gdf: GeoDataFrame
    
    Returns:
        dict: 各方向の角の座標
    """
    if gdf.empty:
        return None
    
    # 最初のジオメトリを取得
    geometry = gdf.geometry.iloc[0]
    
    # ポリゴンの座標を取得
    if hasattr(geometry, 'exterior'):
        coords = list(geometry.exterior.coords)
    else:
        print("❌ ポリゴンジオメトリが見つかりませんでした")
        return None
    
    if len(coords) < 3:
        return None
    
    coords_array = np.array(coords)
    lons = coords_array[:, 0]
    lats = coords_array[:, 1]
    
    # 全体の中心を計算
    center_lon = np.mean(lons)
    center_lat = np.mean(lats)
    
    # 各方向の角を見つける
    corners = {}
    
    # 東側の角（経度最大）
    east_idx = np.argmax(lons)
    corners["東"] = (lons[east_idx], lats[east_idx])
    
    # 西側の角（経度最小）
    west_idx = np.argmin(lons)
    corners["西"] = (lons[west_idx], lats[west_idx])
    
    # 北側の角（緯度最大）
    north_idx = np.argmax(lats)
    corners["北"] = (lons[north_idx], lats[north_idx])
    
    # 南側の角（緯度最小）
    south_idx = np.argmin(lats)
    corners["南"] = (lons[south_idx], lats[south_idx])
    
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
    
    return corners

def find_edge_center_between_corners(coords, corner1, corner2):
    """
    2つの角の間の辺の中央座標を見つける関数
    
    Args:
        coords: 座標のリスト [(lon, lat), ...]
        corner1: 1つ目の角の座標 (lon, lat)
        corner2: 2つ目の角の座標 (lon, lat)
    
    Returns:
        (lon, lat): 辺の中央座標
    """
    if len(coords) < 3:
        return None
    
    coords_array = np.array(coords)
    lons = coords_array[:, 0]
    lats = coords_array[:, 1]
    
    # 2つの角の間の座標をフィルタリング
    # 角1と角2の間にある座標を選択
    edge_coords = []
    
    for i, (lon, lat) in enumerate(coords):
        # 角1と角2の間にあるかチェック
        lon_between = min(corner1[0], corner2[0]) <= lon <= max(corner1[0], corner2[0])
        lat_between = min(corner1[1], corner2[1]) <= lat <= max(corner1[1], corner2[1])
        
        if lon_between and lat_between:
            edge_coords.append((lon, lat, i))
    
    if len(edge_coords) > 0:
        # 中央の座標を選択
        mid_idx = len(edge_coords) // 2
        return (edge_coords[mid_idx][0], edge_coords[mid_idx][1])
    
    # フィルタリングで座標が見つからない場合は、2つの角の中点を返す
    return ((corner1[0] + corner2[0]) / 2, (corner1[1] + corner2[1]) / 2)

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

def calculate_direction_edge_center(gdf, direction):
    """
    GeoDataFrameから指定された方角の辺上の中央座標を計算する関数（隣接角の中央ベース + ポリゴン線上）
    
    Args:
        gdf: GeoDataFrame
        direction: 方角 ("東", "西", "南", "北", "北東", "北西", "南東", "南西")
    
    Returns:
        (lon, lat): 辺上の中央座標（ポリゴンの線上）
    """
    if gdf.empty:
        return None
    
    # 座標を取得
    geometry = gdf.geometry.iloc[0]
    if hasattr(geometry, 'exterior'):
        coords = list(geometry.exterior.coords)
    else:
        return None
    
    # まず角を見つける
    corners = find_corners(gdf)
    if not corners:
        return None
    
    # 方角に基づいて隣接する角の中央を計算し、ポリゴン線上に投影
    if direction == "東":
        # 東側の角を使用（NEとSEの中央）
        if "北東" in corners and "南東" in corners:
            ne_corner = corners["北東"]
            se_corner = corners["南東"]
            center_point = ((ne_corner[0] + se_corner[0]) / 2, (ne_corner[1] + se_corner[1]) / 2)
            return find_closest_point_on_polygon(coords, center_point)
        else:
            return corners.get("東")
    
    elif direction == "西":
        # 西側の角を使用（NWとSWの中央）
        if "北西" in corners and "南西" in corners:
            nw_corner = corners["北西"]
            sw_corner = corners["南西"]
            center_point = ((nw_corner[0] + sw_corner[0]) / 2, (nw_corner[1] + sw_corner[1]) / 2)
            return find_closest_point_on_polygon(coords, center_point)
        else:
            return corners.get("西")
    
    elif direction == "北":
        # 北側の角を使用（NEとNWの中央）
        if "北東" in corners and "北西" in corners:
            ne_corner = corners["北東"]
            nw_corner = corners["北西"]
            center_point = ((ne_corner[0] + nw_corner[0]) / 2, (ne_corner[1] + nw_corner[1]) / 2)
            return find_closest_point_on_polygon(coords, center_point)
        else:
            return corners.get("北")
    
    elif direction == "南":
        # 南側の角を使用（SEとSWの中央）
        if "南東" in corners and "南西" in corners:
            se_corner = corners["南東"]
            sw_corner = corners["南西"]
            center_point = ((se_corner[0] + sw_corner[0]) / 2, (se_corner[1] + sw_corner[1]) / 2)
            return find_closest_point_on_polygon(coords, center_point)
        else:
            return corners.get("南")
    
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
    場所名と方角を指定して、その方角の辺上の中央座標を取得する関数
    
    Args:
        place: 検索する場所名
        direction: 方角 ("東", "西", "南", "北", "北東", "北西", "南東", "南西")
    
    Returns:
        (lon, lat): 辺上の中央座標（ポリゴンの線上）
    """
    # osmnxでデータを取得
    gdf = get_place_polygon(place)
    
    if gdf is not None:
        # 方角別の辺上中央座標を計算
        edge_center = calculate_direction_edge_center(gdf, direction)
        
        if edge_center:
            print(f"📍 {direction}側の辺上中央座標: 経度={edge_center[0]:.6f}, 緯度={edge_center[1]:.6f}")
            return edge_center
        else:
            print(f"❌ {direction}側の辺上中央座標を計算できませんでした")
            return None
    else:
        print(f"❌ 場所 '{place}' のデータを取得できませんでした")
        return None

def visualize_with_direction_centers(place):
    """
    場所名を指定して、全方角の辺上中央座標を可視化する関数
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
        
        # 角を見つけてプロット
        corners = find_corners(gdf)
        if corners:
            corner_lons = [corner[0] for corner in corners.values()]
            corner_lats = [corner[1] for corner in corners.values()]
            plt.plot(corner_lons, corner_lats, 'rx', markersize=15, label='Corners', zorder=12)
        
        # 各方角の辺上中央座標を計算・プロット
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
            edge_center = calculate_direction_edge_center(gdf, direction)
            if edge_center:
                plt.plot(edge_center[0], edge_center[1], 'o', 
                        color=direction_colors[direction], markersize=12, 
                        markeredgecolor='white', markeredgewidth=2,
                        label=f'{direction_labels[i]} Center', zorder=15)
                
                # ラベルを追加（オフセット付き）
                offset = label_offsets[direction]
                label_x = float(edge_center[0]) + offset[0]
                label_y = float(edge_center[1]) + offset[1]
                plt.text(label_x, label_y, direction_labels[i], 
                        ha='center', va='center', fontsize=10, weight='bold', 
                        color=direction_colors[direction], 
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        plt.title(f"Place: {place} Direction Centers on Polygon Line (OSMnx)", fontsize=14)
        plt.xlabel("Longitude", fontsize=12)
        plt.ylabel("Latitude", fontsize=12)
        plt.legend(fontsize=10, loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # ファイルに保存
        output_file = f"place_{place.replace(', ', '_').replace(' ', '_')}_direction_centers_osmnx.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ 可視化結果を保存しました: {output_file}")
        
        plt.show()
        
    else:
        print(f"❌ 場所 '{place}' のデータを取得できませんでした")

# メイン実行部分
if __name__ == "__main__":
    print(f"🔍 場所 '{PLACE}' の座標をosmnxから取得中...")
    print("🔍 角ベースのアルゴリズムで各方向の位置を決定します")
    
    # 全方角の辺上中央座標を計算
    directions = ["東", "西", "南", "北", "北東", "北西", "南東", "南西"]
    
    print("\n📍 各方向の辺上中央座標（角ベースアルゴリズム）:")
    for direction in directions:
        get_location_by_direction(PLACE, direction)
    
    print("\n🎨 可視化を実行中...")
    visualize_with_direction_centers(PLACE)
    
    print("\n💡 osmnxライブラリを使用して高速にデータを取得しました")