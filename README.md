# cs-python

# はじめに
[CS立体図](https://www.pref.nagano.lg.jp/ringyosogo/soshiki/documents/iku_cs_1.pdf)は標高の値と、標高から計算した傾斜と曲率の画像を組み合わせて作成する地形表現図です。
このCS立体図を作成する際において、傾斜や曲率を計算する際に使用する窓サイズや、標高・傾斜・曲率を画像として合成する比率などのパラメータがします。
パラメータを変えると、地形の見え方が変わるため、見たい対象の地形に合わせて最適なパラメータを設定する必要があります。
そこで、このパラメータを調整するためのツールが必要になります。
通常であれば、[QGISのプラグイン](https://github.com/waigania13/CSMapMaker)を使うことが多いですが、今回はPythonベースでやってみたいと思います。
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3792785/cbbc66f5-6639-46d5-aa43-86dd3745c34c.png)

# ベースになる実装
CS立体図を作成するためのライブラリとして、MIERUNEさんが[csmap-py](https://github.com/MIERUNE/csmap-py)を提供してくれています。
今回は、このライブラリで提供されている[傾斜・曲率の計算](https://github.com/MIERUNE/csmap-py/blob/main/csmap/calc.py)、[画像合成の計算](https://github.com/MIERUNE/csmap-py/blob/main/csmap/color.py)を用います。
[傾斜・曲率の計算](https://github.com/MIERUNE/csmap-py/blob/main/csmap/calc.py)と[画像合成の計算](https://github.com/MIERUNE/csmap-py/blob/main/csmap/color.py)をダウンロードして、プロジェクトのフォルダに格納してください。


詳しくは[MIERUNEさんの記事](https://qiita.com/Kanahiro/items/744dee4795800570b01b)を参照ください。

# 下準備
MIERUNEさんのライブラリを呼び出すためにapp.pyと同じフォルダにcolor.pyとcalc.pyを格納してください。
そして、それぞれ以下のように書き換えてください。


```python:calc.py
import numpy as np
import cv2

def slope(dem: np.ndarray, size=3) -> np.ndarray:
    """傾斜率を求める
    出力のndarrayのshapeは、(dem.shape[0] - 2, dem.shape[1] - 2)
    """
    # ぼかしを入れて疑似的に広範囲の情報を用いる
    if(size>1):
        dem = cv2.blur(dem, ksize=(size, size))
    z2 = dem[1:-1, 0:-2]
    z4 = dem[0:-2, 1:-1]
    z6 = dem[2:, 1:-1]
    z8 = dem[1:-1, 2:]
    p = (z6 - z4) / 2
    q = (z8 - z2) / 2
    p2 = p * p
    q2 = q * q

    slope = np.arctan((p2 + q2) ** 0.5)
    return slope


def gaussianfilter(image: np.ndarray, size: int, sigma: int) -> np.ndarray:
    """ガウシアンフィルター"""
    size = int(size) // 2
    x, y = np.mgrid[-size : size + 1, -size : size + 1]
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = g / g.sum()

    # 画像を畳み込む
    k_h, k_w = kernel.shape
    i_h, i_w = image.shape

    # パディングサイズを計算
    pad_h, pad_w = k_h // 2, k_w // 2

    # 画像にパディングを適用
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")

    # einsumを使用して畳み込みを行う
    sub_matrices = np.lib.stride_tricks.as_strided(
        padded, shape=(i_h, i_w, k_h, k_w), strides=padded.strides * 2
    )
    return np.einsum("ijkl,kl->ij", sub_matrices, kernel)


def curvature(dem: np.ndarray, cell_size: int) -> np.ndarray:
    """曲率を求める"""

    # SAGA の Slope, Aspect, Curvature の 9 parameter 2nd order polynom に準拠
    z1 = dem[0:-2, 0:-2]
    z2 = dem[1:-1, 0:-2]
    z3 = dem[2:, 0:-2]
    z4 = dem[0:-2, 1:-1]
    z5 = dem[1:-1, 1:-1]
    z6 = dem[2:, 1:-1]
    z7 = dem[0:-2, 2:]
    z8 = dem[1:-1, 2:]
    z9 = dem[2:, 2:]

    cell_area = cell_size * cell_size
    r = ((z4 + z6) / 2 - z5) / cell_area
    t = ((z2 + z8) / 2 - z5) / cell_area
    p = (z6 - z4) / (2 * cell_size)
    q = (z8 - z2) / (2 * cell_size)
    s = (z1 - z3 - z7 + z9) / (4 * cell_area)
    p2 = p * p
    q2 = q * q
    spq = s * p * q

    # gene
    return -2 * (r + t)

    # plan
    p2_q2 = p2 + q2
    p2_q2 = np.where(p2_q2 > 1e-6, p2_q2, np.nan)
    return -(t * p2 + r * q2 - 2 * spq) / ((p2_q2) ** 1.5)
```
```python:color.py
import numpy as np


def rgbify(arr: np.ndarray, method, scale: (float, float) = None) -> np.ndarray:
    """ndarrayをRGBに変換する
    - arrは変更しない
    - ndarrayのshapeは、(4, height, width) 4はRGBA
    """

    _min = arr.min() if scale is None else scale[0]
    _max = arr.max() if scale is None else scale[1]

    # -x ~ x を 0 ~ 1 に正規化
    arr = (arr - _min) / (_max - _min)
    # clamp
    arr = np.where(arr < 0, 0, arr)
    arr = np.where(arr > 1, 1, arr)

    # 3次元に変換
    if(method == "height_blackwhite"):
        rgb = height_blackwhite(arr)
    elif(method == "slope_red"):
        rgb = slope_red(arr)
    elif(method == "slope_blackwhite"):
        rgb = slope_blackwhite(arr)
    elif(method == "curvature_blue"):
        rgb = curvature_blue(arr)
    elif(method == "curvature_redyellowblue"):
        rgb = curvature_redyellowblue(arr)
    return rgb


def slope_red(arr: np.ndarray) -> np.ndarray:
    rgb = np.zeros((4, arr.shape[0], arr.shape[1]), dtype=np.uint8)
    rgb[0, :, :] = 255 - arr * 155  # R: 255 -> 100
    rgb[1, :, :] = 245 - arr * 195  # G: 245 -> 50
    rgb[2, :, :] = 235 - arr * 215  # B: 235 -> 20
    rgb[3, :, :] = 255
    return rgb


def slope_blackwhite(arr: np.ndarray) -> np.ndarray:
    rgb = np.zeros((4, arr.shape[0], arr.shape[1]), dtype=np.uint8)
    rgb[0, :, :] = (1 - arr) * 255  # R
    rgb[1, :, :] = (1 - arr) * 255  # G
    rgb[2, :, :] = (1 - arr) * 255  # B
    rgb[3, :, :] = 255  # A
    return rgb


def curvature_blue(arr: np.ndarray) -> np.ndarray:
    rgb = np.zeros((4, arr.shape[0], arr.shape[1]), dtype=np.uint8)
    rgb[0, :, :] = 35 + arr * 190  # R: 35 -> 225
    rgb[1, :, :] = 80 + arr * 155  # G: 80 -> 235
    rgb[2, :, :] = 100 + arr * 145  # B: 100 -> 245
    rgb[3, :, :] = 255
    return rgb


def curvature_redyellowblue(arr: np.ndarray) -> np.ndarray:
    # value:0-1 to: red -> yellow -> blue
    # interpolate between red and yellow, and yellow and blue, by linear

    # 0-0.5: blue -> white
    rgb1 = np.zeros((4, arr.shape[0], arr.shape[1]), dtype=np.uint8)
    rgb1[0, :, :] = 75 + arr * 170 * 2  # R: 75 -> 245
    rgb1[1, :, :] = 100 + arr * 145 * 2  # G: 100 -> 245
    rgb1[2, :, :] = 165 + arr * 80 * 2  # B: 165 -> 245

    # 0.5-1: white -> red
    rgb2 = np.zeros((4, arr.shape[0], arr.shape[1]), dtype=np.uint8)
    rgb2[0, :, :] = 245 - (arr * 2 - 1) * 100  # R: 245 -> 145
    rgb2[1, :, :] = 245 - (arr * 2 - 1) * 190  # G: 245 -> 55
    rgb2[2, :, :] = 245 - (arr * 2 - 1) * 195  # B: 245 -> 50

    # blend
    rgb = np.where(arr < 0.5, rgb1, rgb2)
    rgb[3, :, :] = 255

    return rgb


def height_blackwhite(arr: np.ndarray) -> np.ndarray:
    rgb = np.zeros((4, arr.shape[0], arr.shape[1]), dtype=np.uint8)
    rgb[0, :, :] = (1 - arr) * 255  # R
    rgb[1, :, :] = (1 - arr) * 255  # G
    rgb[2, :, :] = (1 - arr) * 255  # B
    rgb[3, :, :] = 255
    return rgb


def blend(
    dem_bw: np.ndarray,
    slope_red: np.ndarray,
    slope_bw: np.ndarray,
    curvature_blue: np.ndarray,
    curvature_ryb: np.ndarray,
    blend_params: dict = {
        "slope_bw": 0.5,  # alpha blending based on the paper
        "curvature_ryb": 0.25,  # 0.5 / 2
        "slope_red": 0.125,  # 0.5 / 2 / 2
        "curvature_blue": 0.06125,  # 0.5 / 2 / 2 / 2
        "dem": 0.030625,  # 0.5 / 2 / 2 / 2 / 2
    },
) -> np.ndarray:
    """blend all rgb
    全てのndarrayは同じshapeであること
    DEMを用いて処理した他の要素は、DEMよりも1px内側にpaddingされているので
    あらかじめDEMのpaddingを除外しておく必要がある
    """
    _blend = np.zeros((4, dem_bw.shape[0], dem_bw.shape[1]), dtype=np.uint8)
    _blend = (
        dem_bw * blend_params["dem"]
        + slope_red * blend_params["slope_red"]
        + slope_bw * blend_params["slope_bw"]
        + curvature_blue * blend_params["curvature_blue"]
        + curvature_ryb * blend_params["curvature_ryb"]
    )
    _blend = _blend.astype(np.uint8)  # force uint8
    _blend[3, :, :] = 255  # alpha
    return _blend

```

# ブラウザでの表示
ブラウザで動作するパラメータを調整するGUIを作ります。
python向けのライブラリで、Streamlitがあります。
ブラウザ表示部分はこのように書きます。
実行する際には、`streamlit run app.py`とコマンドを端末に打ち込むと、WebブラウザにGUIが立ち上がります。

```python:app.py
import streamlit as st
from PIL import Image
from calc import *
from color import *
from tifffile import TiffFile
st.title("CS立体図作成")

blend_params: dict = {
        "slope_bw": 0.5,  # alpha blending based on the paper
        "curvature_ryb": 0.25,  # 0.5 / 2
        "slope_red": 0.125,  # 0.5 / 2 / 2
        "curvature_blue": 0.06125,  # 0.5 / 2 / 2 / 2
        "dem": 0.030625,  # 0.5 / 2 / 2 / 2 / 2
    }
    
class CsmapParams:
    gf_size: int = 12
    gf_sigma: int = 3
    slope_size: int = 1
    curvature_size: int = 1
    height_scale: (float, float) = (0.0, 3000.0)
    slope_scale: (float, float) = (0.0, 1.5)
    curvature_scale: (float, float) = (-0.1, 0.1)

# tifを扱う
file_path = st.file_uploader('', type=['tif'])
if file_path :
    with TiffFile(file_path) as tif:
        img = tif.asarray()[:,:]
    
    # 傾斜計算のパラメータ
    # expand_slope = st.sidebar.checkbox("傾斜")
    slope_size = st.sidebar.slider("傾斜", min_value=1, max_value=13, step=1, value=1)
    # 曲率計算のパラメータ
    # expand_curve = st.sidebar.checkbox("曲率")
    curvature_size = st.sidebar.slider("曲率", min_value=1, max_value=13, step=1, value=3)
    # ガウシアンのパラメータ
    gauss_size = st.sidebar.slider("フィルタサイズ", min_value=3, max_value=21, step=1, value=13)
    # 画像化のパラメータ
    height_scale_min = st.sidebar.slider("高さの最小値", min_value=0, max_value=3776, step=100, value=0)
    height_scale_max = st.sidebar.slider("高さの最大値", min_value=0, max_value=3776, step=10, value=1500)

    # 画像合成のパラメータ
    slope_bw_param = st.sidebar.slider("傾斜(グレー)の混合率", min_value=0, max_value=100, step=1, value=50)
    curvature_ryb_param = st.sidebar.slider("曲率(赤-青)の混合率", min_value=0, max_value=100, step=1, value=25)  
    slope_red_param = st.sidebar.slider("傾斜(赤)の混合率", min_value=0, max_value=100, step=1, value=12)
    curvature_blue_param = st.sidebar.slider("曲率(青)の混合率", min_value=0, max_value=100, step=1, value=6)  
    dem_base_param = st.sidebar.slider("DEMの混合率", min_value=0, max_value=100, step=1, value=3)  
    # 読み込んだ画像に対する前処理
    params = CsmapParams()
    # 傾斜
    params.slope_size = slope_size
    slope_raw = slope(img, size=params.slope_size)

    # 曲率 
    params.curvature_size = curvature_size
    curvature_raw = curvature(img, params.curvature_size)

    #ガウシアン
    params.gf_size = gauss_size
    gaussian_raw = gaussianfilter(img, params.gf_size, params.gf_sigma)

    # RGB 変換
    params.height_scale = (height_scale_min, height_scale_max)
    dem_rgb =  rgbify(img, "height_blackwhite", scale=params.height_scale)

    # RGB変換
    slope_red= rgbify(slope_raw, "slope_red", scale=params.slope_scale)
    slope_bw = rgbify(slope_raw, "slope_blackwhite", scale=params.slope_scale)

    # RGBへ変換
    curvature_blue = rgbify(
        curvature_raw, "curvature_blue", scale=params.curvature_scale
    )
    curvature_ryb = rgbify(
        curvature_raw, "curvature_redyellowblue", scale=params.curvature_scale
    )


    dem_rgb = dem_rgb[:, 1:-1, 1:-1]  # remove padding

    # 混ぜる
    # 混ぜる割合
    slope_bw_param = slope_bw_param/100.0
    curvature_ryb_param = curvature_ryb_param/100.0
    slope_red_param = slope_red_param/100.0
    curvature_blue_param = curvature_blue_param/100.0
    dem_base_param = dem_base_param/100.0
    # デフォルト値
    # "slope_bw": 0.5,  # alpha blending based on the paper
    # "curvature_ryb": 0.25,  # 0.5 / 2
    # "slope_red": 0.125,  # 0.5 / 2 / 2
    # "curvature_blue": 0.06125,  # 0.5 / 2 / 2 / 2
    # "dem": 0.030625,  # 0.5 / 2 / 2 / 2 / 2
    blend_params: dict = {
        "slope_bw": slope_bw_param,  # alpha blending based on the paper
        "curvature_ryb": curvature_ryb_param,  # 0.5 / 2
        "slope_red": slope_red_param,  # 0.5 / 2 / 2
        "curvature_blue": curvature_blue_param,  # 0.5 / 2 / 2 / 2
        "dem": dem_base_param,  # 0.5 / 2 / 2 / 2 / 2
    }
    # blend all rgb
    blend_rgb = blend(
        dem_rgb,
        slope_red,
        slope_bw,
        curvature_blue,
        curvature_ryb,
        blend_params
    )
    # 4x縦x横なので配列を変形させる
    pil_image = Image.fromarray(blend_rgb.transpose(1,2,0)[:,:,:3])


    # 結果を表示  
    st.image(pil_image)
```


    
