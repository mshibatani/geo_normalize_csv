# 方向マーカー配置アルゴリズム仕様書 📍

## 📝 このドキュメントについて

このドキュメントは、方向マーカー配置アルゴリズムの仕様を定義するものです。

### 開発者への指示

1. **仕様書への準拠**：
   - このドキュメントに記載されたアルゴリズムと実装方法に従って開発を行ってください
   - 特に南北方向（N・S）の分離については、記載された垂直投影アルゴリズムを厳守してください
   - 実装時に不明点がある場合は、まずこの仕様書を参照してください

2. **仕様書の問題点の指摘**：
   - この仕様書自体に問題点や改善点を発見した場合は、遠慮なく指摘してください
   - 指摘の際は具体的な問題点と、可能であれば修正案を提示してください
   - 特に方向マーカーが正しく分離されない場合や、予期しない位置に配置される場合は、すぐに報告してください

3. **改善提案**：
   - アルゴリズムの効率化や精度向上のための提案も歓迎します
   - 提案は「〜の部分を〜のように変更することで、〜の問題が解決できる」という形式で具体的に記述してください

### 変更履歴

- 2024-08-05: 初版作成
- 2024-08-05: 南側（S）の配置要件を明確化 - SW-SE間のポリゴン線上に配置することを明記
- 2024-08-05: ポリゴンの実形状の尊重を基本要件に追加 - 対角点を直線で結んだ線上ではなく、ポリゴンの実際の辺上にマーカーを配置する原則を明確化
- 2024-08-05: ポリゴン辺取得アルゴリズムを追加 - `find_polygon_edge_between_points`関数の仕様を追加
- [ここに今後の変更履歴を記録]

## 🎯 目的

地図上で場所の輪郭（ポリゴン）から方向マーカー（N・E・S・W、NE・NW・SE・SW）を正確に配置するためのアルゴリズム仕様です。このドキュメントは、`get_outline_location.py` で実装されている方向マーカー配置ロジックの要件と実装詳細を説明します。

## 📊 方向マーカーの配置要件

### 基本要件

1. **8方向のマーカー配置**：
   - 4つの主要方向（N・E・S・W）
   - 4つの対角方向（NE・NW・SE・SW）

2. **マーカー位置の条件**：
   - **対角方向（NE・NW・SE・SW）**：ポリゴンの外接矩形の四隅に配置
   - **主要方向（N・E・S・W）**：対応する辺の中央付近でポリゴン線上に配置
     - **N**：NW-NE間のポリゴン線上（北側の辺）
     - **E**：NE-SE間のポリゴン線上（東側の辺）
     - **S**：SW-SE間のポリゴン線上（南側の辺）- 重要：SE-NE間ではなく
     - **W**：NW-SW間のポリゴン線上（西側の辺）

3. **マーカーの重複防止**：
   - 各マーカーは明確に分離されていること
   - 特にN/S、E/Wが重複しないこと

4. **ポリゴンの実形状の尊重**：
   - 主要方向（N・E・S・W）の配置は、対角点を直線で結んだ線上ではなく、**ポリゴンの実際の辺上**に配置すること
   - ポリゴンの形状が複雑な場合でも、実際の辺の形状に沿ってマーカーを配置すること
   - 実際の辺が複数の点で構成される場合は、その辺の中点を使用すること

## 🔧 実装アルゴリズム

### 1. 対角方向（NE・NW・SE・SW）の決定

```python
# 回転後の座標系で四隅を決定
x_min, x_max = min(rotated_lons), max(rotated_lons)
y_min, y_max = min(rotated_lats), max(rotated_lats)

# 各象限の座標点を見つける
for i, (lon, lat) in enumerate(rotated_coords):
    # 北東 (NE): X大きい、Y大きい
    if lon > x_center and lat > y_center:
        if "北東" not in corners or lat > corners["北東"][1]:
            corners["北東"] = (lon, lat)
    # 北西 (NW): X小さい、Y大きい
    elif lon < x_center and lat > y_center:
        if "北西" not in corners or lat > corners["北西"][1]:
            corners["北西"] = (lon, lat)
    # 南東 (SE): X大きい、Y小さい
    elif lon > x_center and lat < y_center:
        if "南東" not in corners or lon > corners["南東"][0]:
            corners["南東"] = (lon, lat)
    # 南西 (SW): X小さい、Y小さい
    elif lon < x_center and lat < y_center:
        if "南西" not in corners or lon < corners["南西"][0]:
            corners["南西"] = (lon, lat)
```

### 2. 主要方向（N・E・S・W）の決定

#### 東西方向（E・W）

```python
# E: NE-SE間の線分中点
if "北東" in corners and "南東" in corners:
    ne_corner = corners["北東"]
    se_corner = corners["南東"]
    
    # ポリゴン上のNE-SE間の実際の辺を取得
    ne_se_edge = find_polygon_edge_between_points(coords, ne_corner, se_corner)
    
    if ne_se_edge and len(ne_se_edge) > 1:
        # 辺の中点を計算
        edge_midpoint_idx = len(ne_se_edge) // 2
        if len(ne_se_edge) % 2 == 0:  # 偶数個の点がある場合
            p1 = ne_se_edge[edge_midpoint_idx - 1]
            p2 = ne_se_edge[edge_midpoint_idx]
            center_point = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        else:  # 奇数個の点がある場合
            center_point = ne_se_edge[edge_midpoint_idx]
        
        corners["東"] = center_point
    else:
        # 辺が見つからない場合は従来の方法でフォールバック
        center_point = ((ne_corner[0] + se_corner[0]) / 2, (ne_corner[1] + se_corner[1]) / 2)
        corners["東"] = find_perpendicular_projection_on_edge(ne_corner, se_corner, center_point, coords, direction_hint="east")

# W: NW-SW間の線分中点
if "北西" in corners and "南西" in corners:
    nw_corner = corners["北西"]
    sw_corner = corners["南西"]
    
    # ポリゴン上のNW-SW間の実際の辺を取得
    nw_sw_edge = find_polygon_edge_between_points(coords, nw_corner, sw_corner)
    
    if nw_sw_edge and len(nw_sw_edge) > 1:
        # 辺の中点を計算
        edge_midpoint_idx = len(nw_sw_edge) // 2
        if len(nw_sw_edge) % 2 == 0:  # 偶数個の点がある場合
            p1 = nw_sw_edge[edge_midpoint_idx - 1]
            p2 = nw_sw_edge[edge_midpoint_idx]
            center_point = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        else:  # 奇数個の点がある場合
            center_point = nw_sw_edge[edge_midpoint_idx]
        
        corners["西"] = center_point
    else:
        # 辺が見つからない場合は従来の方法でフォールバック
        center_point = ((nw_corner[0] + sw_corner[0]) / 2, (nw_corner[1] + sw_corner[1]) / 2)
        corners["西"] = find_perpendicular_projection_on_edge(nw_corner, sw_corner, center_point, coords, direction_hint="west")
```

#### 南北方向（N・S）

```python
# N: NW-NE間の線上で、中点から垂直に引いた線との交点
if "北東" in corners and "北西" in corners:
    nw_corner = corners["北西"]
    ne_corner = corners["北東"]
    
    # ポリゴン上のNW-NE間の実際の辺を取得
    nw_ne_edge = find_polygon_edge_between_points(coords, nw_corner, ne_corner)
    
    if nw_ne_edge and len(nw_ne_edge) > 1:
        # 辺の中点を計算
        edge_midpoint_idx = len(nw_ne_edge) // 2
        if len(nw_ne_edge) % 2 == 0:  # 偶数個の点がある場合
            p1 = nw_ne_edge[edge_midpoint_idx - 1]
            p2 = nw_ne_edge[edge_midpoint_idx]
            center_point = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        else:  # 奇数個の点がある場合
            center_point = nw_ne_edge[edge_midpoint_idx]
        
        corners["北"] = center_point
    else:
        # 辺が見つからない場合は従来の方法でフォールバック
        center_point = ((nw_corner[0] + ne_corner[0]) / 2, (nw_corner[1] + ne_corner[1]) / 2)
        corners["北"] = find_perpendicular_projection_on_edge(nw_corner, ne_corner, center_point, coords, direction_hint="north")

# S: SW-SE間の線上で、中点から垂直に引いた線との交点
# 重要: Sは必ずSW-SE間のポリゴン線上に配置すること
# EとSが重複しないよう、SE-NE間ではなくSW-SE間を使用する
if "南東" in corners and "南西" in corners:
    sw_corner = corners["南西"]
    se_corner = corners["南東"]
    
    # ポリゴン上のSW-SE間の実際の辺を取得
    sw_se_edge = find_polygon_edge_between_points(coords, sw_corner, se_corner)
    
    if sw_se_edge and len(sw_se_edge) > 1:
        # 辺の中点を計算
        edge_midpoint_idx = len(sw_se_edge) // 2
        if len(sw_se_edge) % 2 == 0:  # 偶数個の点がある場合
            p1 = sw_se_edge[edge_midpoint_idx - 1]
            p2 = sw_se_edge[edge_midpoint_idx]
            center_point = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        else:  # 奇数個の点がある場合
            center_point = sw_se_edge[edge_midpoint_idx]
        
        corners["南"] = center_point
    else:
        # 辺が見つからない場合は従来の方法でフォールバック
        center_point = ((sw_corner[0] + se_corner[0]) / 2, (sw_corner[1] + se_corner[1]) / 2)
        corners["南"] = find_perpendicular_projection_on_edge(sw_corner, se_corner, center_point, coords, direction_hint="south")
```

### 3. ポリゴン辺取得アルゴリズム

```python
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
    
    return edge_points
```

### 4. 垂直投影アルゴリズム

```python
def find_perpendicular_projection_on_edge(corner1, corner2, target_point, coords, direction_hint=None):
    """
    真の垂直投影を計算する関数
    """
    # EまたはWの場合：線分の中点を直接計算
    if direction_hint in ["east", "west"]:
        midpoint = find_line_segment_midpoint(corner1, corner2)
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
            perpendicular_vector = (edge_vector[1], -edge_vector[0])
            # ベクトルの長さを正規化して方向を強調
            length = np.sqrt(perpendicular_vector[0]**2 + perpendicular_vector[1]**2)
            if length > 0:
                perpendicular_vector = (perpendicular_vector[0]/length * 0.001, perpendicular_vector[1]/length * 0.005)
        
        # 垂直線とポリゴンの交点を見つける
        intersection = find_perpendicular_intersection_with_polygon(midpoint, perpendicular_vector, coords)
        
        if intersection:
            return intersection
        else:
            # 交点が見つからない場合は、方向に応じて別の方法を試す
            # ポリゴン上の点を探す（方向を考慮）
            coords_array = np.array(coords)
            
            if direction_hint == "north":
                # 北側：Y座標が大きい点を優先
                y_sorted = np.argsort(coords_array[:, 1])[::-1]  # Y座標の降順
                for idx in y_sorted[:5]:  # 上位5点を検討
                    point = coords_array[idx]
                    # X座標が近いかチェック
                    if abs(point[0] - midpoint[0]) < 0.0005:
                        return tuple(point)
            
            elif direction_hint == "south":
                # 南側：Y座標が小さい点を優先
                y_sorted = np.argsort(coords_array[:, 1])  # Y座標の昇順
                for idx in y_sorted[:5]:  # 上位5点を検討
                    point = coords_array[idx]
                    # X座標が近いかチェック
                    if abs(point[0] - midpoint[0]) < 0.0005:
                        return tuple(point)
            
            # それでも見つからない場合は中点を使用
            return midpoint
    
    # その他の場合：中点を返す
    midpoint = find_line_segment_midpoint(corner1, corner2)
    return midpoint
```

## 🚨 実装上の注意点

1. **方向ベクトルの強調**：
   - 北向きと南向きのベクトルは明確に分離する必要がある
   - Y方向の係数を大きく（0.005）、X方向の係数を小さく（0.001）設定して南北方向を強調

2. **交点が見つからない場合の対応**：
   - 北側はY座標が大きい点を優先
   - 南側はY座標が小さい点を優先
   - X座標が中点に近い点を選択する

3. **重複防止**：
   - 南北方向は特に重複しやすいため、垂直ベクトルの方向を明確に分ける
   - 交点が見つからない場合は方向に応じた代替アルゴリズムを使用

## 📊 期待される結果

1. **対角方向（NE・NW・SE・SW）**：
   - ポリゴンの外接矩形の四隅に配置される
   - 各象限に1つずつ、明確に分離されている

2. **主要方向（N・E・S・W）**：
   - 対応する辺の中央付近でポリゴン線上に配置される
   - 特にN/S、E/Wは明確に分離されている
   - 東西方向は単純な中点、南北方向は垂直投影による交点

## 🛠️ デバッグ方法

問題が発生した場合は、以下の点を確認してください：

1. **座標の出力**：
   ```python
   print(f"🔍 デバッグ: 東 = ({corners['東'][0]:.6f}, {corners['東'][1]:.6f})")
   print(f"🔍 デバッグ: 西 = ({corners['西'][0]:.6f}, {corners['西'][1]:.6f})")
   print(f"🔍 デバッグ: 北 = ({corners['北'][0]:.6f}, {corners['北'][1]:.6f})")
   print(f"🔍 デバッグ: 南 = ({corners['南'][0]:.6f}, {corners['南'][1]:.6f})")
   ```

2. **垂直投影の確認**：
   ```python
   print(f"🔍 垂直投影デバッグ: 角1={corner1}, 角2={corner2}, 方向={direction_hint}")
   print(f"🔍 南北方向：中点={midpoint}, 垂直ベクトル={perpendicular_vector}")
   ```

3. **可視化範囲の確認**：
   ```python
   print(f"🔍 デバッグ: 可視化範囲 - X: {x_min:.6f} ～ {x_max:.6f}, Y: {y_min:.6f} ～ {y_max:.6f}")
   ```

## 📝 まとめ

この方向マーカー配置アルゴリズムは、地図上のポリゴンから8方向（N・E・S・W・NE・NW・SE・SW）のマーカーを正確に配置します。特に南北方向（N・S）は垂直投影による交点計算と、方向に応じた代替アルゴリズムを組み合わせることで、重複を防止し明確に分離された配置を実現しています。

## 🔄 フィードバックと改善提案

### フィードバックの提出方法

仕様書の問題点や改善提案がある場合は、以下のいずれかの方法でフィードバックを提出してください：

1. **GitHub Issue**: 
   - リポジトリの Issues セクションに新しい Issue を作成
   - タイトルに「[方向マーカー仕様書] 〜について」と記載
   - 問題点や改善提案を具体的に記述

2. **プルリクエスト**:
   - 修正案がある場合は、変更を含むプルリクエストを作成
   - 変更内容と理由を明記

3. **直接コミュニケーション**:
   - チーム内の定例ミーティングで議題として提案
   - 緊急性が高い場合はチャットツールで共有

### フィードバック時に含めるべき情報

1. **問題の具体的な説明**:
   - どのような状況で問題が発生するか
   - 期待される動作と実際の動作の違い
   - 再現手順（可能であれば）

2. **改善提案の場合**:
   - 現在の実装の課題点
   - 提案する変更内容
   - 期待される改善効果

3. **視覚資料**:
   - 問題を示すスクリーンショットや図
   - 修正案を説明する図やコード例

### フィードバック対応プロセス

1. フィードバックを受け取った後、チーム内で検討
2. 仕様書の更新が必要な場合は、変更履歴に記録
3. 実装の変更が必要な場合は、コードも同時に更新
4. 変更後、テストを実施して効果を確認

皆様からのフィードバックを歓迎し、より良いアルゴリズムの実現に向けて協力していきましょう。