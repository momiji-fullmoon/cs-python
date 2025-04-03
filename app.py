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
