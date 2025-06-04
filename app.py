import os
import math
import uuid
import numpy as np
import onnxruntime as ort
from PIL import Image
import gradio as gr
import tempfile

# ——————————————————————————————————————————————————————————————
# 1) Path ke ONNX model “×4”
# ——————————————————————————————————————————————————————————————
MODEL_DIR     = "model"
MODEL_X4_PATH = os.path.join(MODEL_DIR, "Real-ESRGAN-x4plus.onnx")
if not os.path.isfile(MODEL_X4_PATH):
    raise FileNotFoundError(f"ONNX model ×4 tidak ditemukan: {MODEL_X4_PATH}")

# ——————————————————————————————————————————————————————————————
# 2) Buat ONNXRuntime session hanya dengan CPU, 2 thread
# ——————————————————————————————————————————————————————————————
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 2
sess_opts.inter_op_num_threads = 2

session_x4 = ort.InferenceSession(MODEL_X4_PATH, sess_options=sess_opts,
                                  providers=["CPUExecutionProvider"])

# Ambil metadata ukuran input tile untuk model ×4 (biasanya 128×128)
input_meta_x4 = session_x4.get_inputs()[0]
_, _, H_in_x4, W_in_x4 = tuple(input_meta_x4.shape)
H_in_x4, W_in_x4 = int(H_in_x4), int(W_in_x4)

# Cek skala (harusnya 4×)
dummy = np.zeros((1, 3, H_in_x4, W_in_x4), dtype=np.float32)
dummy_out = session_x4.run(None, {input_meta_x4.name: dummy})[0]
_, _, H_out_x4, W_out_x4 = dummy_out.shape
SCALE_X4 = H_out_x4 // H_in_x4
if SCALE_X4 != 4:
    raise RuntimeError(f"Model ×4 menghasilkan scale = {SCALE_X4}, bukan 4")

# ——————————————————————————————————————————————————————————————
# 3) Fungsi util untuk mem‐proses satu tile 128×128 → 512×512
# ——————————————————————————————————————————————————————————————
def run_tile_x4(tile_np: np.ndarray) -> np.ndarray:
    """
    Input: tile_np (128,128,3) float32 ∈ [0,1]
    Output: (512,512,3) float32 ∈ [0,1]
    """
    patch_nchw = np.transpose(tile_np, (2, 0, 1))[None, ...]  # → (1,3,128,128)
    out_nchw = session_x4.run(None, {input_meta_x4.name: patch_nchw})[0]  # (1,3,512,512)
    out_nchw = np.squeeze(out_nchw, axis=0)            # (3,512,512)
    out_hwc  = np.transpose(out_nchw, (1, 2, 0))        # (512,512,3)
    return out_hwc  # float32 ∈ [0,1]

# ——————————————————————————————————————————————————————————————
# 4) Pipeline tile‐based untuk ×4 –– mem‐pad, potong, infer, reassemble, crop
# ——————————————————————————————————————————————————————————————
def tile_upscale_x4(input_img: Image.Image):
    """
    1. PIL→float32 np ∈ [0,1]
    2. Pad agar (H,W) kelipatan H_in_x4 (128)
    3. Bagi menjadi tile (H_in_x4 × W_in_x4)
    4. run_tile_x4 per tile, simpan ke out_arr
    5. Crop ke (orig_h*4, orig_w*4)
    6. Convert ke uint8 PIL, save PNG, return (PIL, filepath)
    """
    # 4.1. Convert PIL→float32 [0,1]
    img_rgb = input_img.convert("RGB")
    arr = np.array(img_rgb).astype(np.float32) / 255.0  # (h_orig, w_orig, 3)
    h_orig, w_orig, _ = arr.shape

    # 4.2. Hitung jumlah tile dan padding
    tiles_h = math.ceil(h_orig / H_in_x4)
    tiles_w = math.ceil(w_orig / W_in_x4)
    pad_h   = tiles_h * H_in_x4 - h_orig
    pad_w   = tiles_w * W_in_x4 - w_orig

    # Reflect pad di kanan + bawah
    arr_padded = np.pad(
        arr,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="reflect"
    )  # → (tiles_h * 128, tiles_w * 128, 3)

    # 4.3. Buat array output berukuran (tiles_h*512, tiles_w*512, 3)
    out_h = tiles_h * H_in_x4 * SCALE_X4
    out_w = tiles_w * W_in_x4 * SCALE_X4
    out_arr = np.zeros((out_h, out_w, 3), dtype=np.float32)

    # 4.4. Loop semua tile
    for i in range(tiles_h):
        for j in range(tiles_w):
            y0   = i * H_in_x4
            x0   = j * W_in_x4
            tile = arr_padded[y0 : y0 + H_in_x4, x0 : x0 + W_in_x4, :]  # (128,128,3)

            up_tile = run_tile_x4(tile)  # (512,512,3) float32

            oy0 = i * H_in_x4 * SCALE_X4
            ox0 = j * W_in_x4 * SCALE_X4
            out_arr[oy0 : oy0 + H_in_x4 * SCALE_X4,
                    ox0 : ox0 + W_in_x4 * SCALE_X4, :] = up_tile

    # 4.5. Crop → (h_orig*4, w_orig*4)
    final_arr = out_arr[0 : h_orig * SCALE_X4, 0 : w_orig * SCALE_X4, :]
    final_arr = np.clip(final_arr, 0.0, 1.0)
    final_uint8 = (final_arr * 255.0).round().astype(np.uint8)
    final_pil   = Image.fromarray(final_uint8)  # (h_orig*4, w_orig*4)

    # 4.6. Simpan ke file PNG unik
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    final_pil.save(tmp.name, format="PNG")
    tmp.close()
    return final_pil, tmp.name

# ——————————————————————————————————————————————————————————————
# 5) Fungsi Standard (×4) dan Premium (×8) untuk Gradio
# ——————————————————————————————————————————————————————————————
def standard_upscale(input_img: Image.Image):
    # Hasil ×4 tile→reassemble 
    return tile_upscale_x4(input_img)

def premium_upscale(input_img: Image.Image):
    """
    - Jalankan pipeline ×4 untuk dapat final_4x (PIL)
    - Lalu bicubic‐resize final_4x → 8× resolusi (orig_w*8, orig_h*8)
    - Simpan PNG baru dan return (PIL_8x, filepath)
    """
    # 5.1. Pertama dapatkan output ×4
    final_4x, path_4x = tile_upscale_x4(input_img)  # final_4x size = (h_orig*4, w_orig*4)

    # 5.2. Ukuran asli
    w_orig, h_orig = input_img.size

    # 5.3. Resize bicubic (LANCZOS) dari 4× → 8×
    target_size = (w_orig * 8, h_orig * 8)
    final_8x = final_4x.resize(target_size, resample=Image.LANCZOS)

    # 5.4. Simpan PNG unik
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    final_8x.save(tmp.name, format="PNG")
    tmp.close()

    return final_8x, tmp.name

# ——————————————————————————————————————————————————————————————
# 6) CSS Kustom agar tombol Premium bergenre “emas”
# ——————————————————————————————————————————————————————————————
css = """
#premium-btn {
    background-color: gold !important;
    color: black !important;
}
"""

# ——————————————————————————————————————————————————————————————
# 7) Bangun Gradio Blocks – 2 tombol berdampingan
# ——————————————————————————————————————————————————————————————
with gr.Blocks(css=css, title="Real-ESRGAN Dual-Mode Upscaler") as demo:
    gr.Markdown(
        """
        # Real-ESRGAN Dual-Mode Upscaler  
        **Standard Upscale** (×4) atau **Premium Upscale 🚀** (×8).  
        Silakan upload gambar apa saja, kemudian klik tombol yang diinginkan.
        """
    )

    # Baris untuk upload gambar
    with gr.Row():
        inp_image = gr.Image(type="pil", label="Upload Source Image")

    # Baris untuk dua tombol
    with gr.Row():
        btn_std  = gr.Button("Standard Upscale (×4)", variant="primary", elem_id="std-btn")
        btn_prem = gr.Button("Premium Upscale 🚀 (×8)", elem_id="premium-btn")

    # Dua output: preview image & link “Download PNG”
    out_preview  = gr.Image(type="pil", label="Upscaled Preview")
    out_download = gr.DownloadButton("⬇️ Download PNG", visible=True)

    # Hubungkan tombol ke fungsi:
    btn_std.click(fn=standard_upscale, inputs=inp_image, outputs=[out_preview, out_download])
    btn_prem.click(fn=premium_upscale, inputs=inp_image, outputs=[out_preview, out_download])

demo.launch(server_name="0.0.0.0", server_port=7860)
