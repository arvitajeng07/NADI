# ============================================================
# NADI (RK4) — FINAL VERSION STABLE (Bagian 1/4)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import soundfile as sf
from datetime import datetime, timedelta
import streamlit.components.v1 as components
import time
import os
import re

# ============================================================
# STATISTIK PENGUNJUNG
# ============================================================

STATS_FILE = "stats.txt"

def init_stats():
    if not os.path.exists(STATS_FILE):
        with open(STATS_FILE, "w") as f:
            f.write("visitors=0\nanalyses=0")

def read_stats():
    with open(STATS_FILE, "r") as f:
        data = f.read().splitlines()
    stats = {}
    for line in data:
        key, value = line.split("=")
        stats[key] = int(value)
    return stats

def write_stats(stats):
    with open(STATS_FILE, "w") as f:
        for k, v in stats.items():
            f.write(f"{k}={v}\n")

init_stats()

if "visitor_logged" not in st.session_state:
    stats = read_stats()
    stats["visitors"] += 1
    write_stats(stats)
    st.session_state.visitor_logged = True

# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(page_title="NADI (RK4) — Stable Final", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "beranda"
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_context" not in st.session_state:
    st.session_state.last_context = None

# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: Inter, sans-serif;
    background: linear-gradient(180deg, #f3f9ff, #eaf6ff);
}

.big-nadi-title {
    font-size: 72px;
    font-weight: 800;
    color: white;
    text-align: center;
    margin-top: -10px;
    text-shadow: 0 0 15px rgba(0,0,0,0.45);
}

.nadi-desc {
    text-align:center;
    color:#6b7280;
    font-size:16px;
    max-width:860px;
    margin:auto;
}

.glass {
    background: rgba(255,255,255,0.88);
    border-radius:14px;
    padding:18px;
    border:1px solid rgba(255,255,255,0.6);
    box-shadow:0 8px 28px rgba(11,99,217,0.08);
    transition:0.2s;
}
.glass:hover {
    transform: translateY(-6px);
    box-shadow:0 20px 42px rgba(11,99,217,0.16);
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# AUDIO HELPERS
# ============================================================

def generate_siren_wav(duration=1.0, sr=44100):
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    mod = 0.5 * (1 + np.sin(2*np.pi*2.2*t))
    freq = 700 + 600*np.sin(2*np.pi*1.0*t)
    tone = 0.9*np.sin(2*np.pi*freq*t) * (0.6 + 0.4*mod)
    tone *= np.linspace(1,0.01,len(t))
    buf = BytesIO()
    sf.write(buf, tone, sr, format="WAV")
    buf.seek(0)
    return buf.read()

def generate_ting_wav(duration=0.45, sr=44100):
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    tone1 = 0.7*np.sin(2*np.pi*1400*t)*np.linspace(1,0,len(t))
    tone2 = 0.5*np.sin(2*np.pi*1800*t)*np.linspace(1,0,len(t))
    tone = tone1 + 0.6*tone2
    buf = BytesIO()
    sf.write(buf, tone, sr, format="WAV")
    buf.seek(0)
    return buf.read()

def wav_bytes_to_datauri(wav_bytes):
    return "data:audio/wav;base64," + base64.b64encode(wav_bytes).decode()

# ============================================================
# RK4
# ============================================================

def rk4_predict_value(last, prev, h=1.0):
    slope = last - prev
    f = lambda t, y: slope
    k1 = f(0, last)
    k2 = f(h/2, last + h*k1/2)
    k3 = f(h/2, last + h*k2/2)
    k4 = f(h, last + h*k3)
    return last + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

def rk4_predict_series(arr):
    arr = [x for x in arr if not pd.isna(x)]
    if len(arr) < 2:
        return None
    return rk4_predict_value(arr[-1], arr[-2])

# ============================================================
# PARSER TENSI GABUNGAN (120/80, 120-80, 120 80)
# ============================================================

def parse_systolic_diastolic_from_cell(cell):
    if pd.isna(cell):
        return None
    s = str(cell).strip()

    pat1 = re.search(r'(\d{2,3})\s*[\/\-\;\,]\s*(\d{2,3})', s)
    if pat1:
        return int(pat1.group(1)), int(pat1.group(2))

    pat2 = re.search(r'^(\d{2,3})\s+(\d{2,3})$', s)
    if pat2:
        return int(pat2.group(1)), int(pat2.group(2))

    return None

# ============================================================
# AUTO-DETECT COLUMNS (ULTRA FLEX)
# ============================================================

def autodetect_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Hapus baris kosong total
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("File kosong setelah pembersihan.")

    # Deteksi kolom gabungan
    parsed = False
    for c in df.columns:
        vals = df[c].dropna().astype(str).head(10).tolist()
        matches = [1 for v in vals if parse_systolic_diastolic_from_cell(v)]
        if len(matches) >= max(1, len(vals)//2):
            s_list = []
            d_list = []
            for v in df[c]:
                p = parse_systolic_diastolic_from_cell(v)
                if p:
                    s_list.append(p[0])
                    d_list.append(p[1])
                else:
                    s_list.append(np.nan)
                    d_list.append(np.nan)
            df[c + "_Systolic"] = s_list
            df[c + "_Diastolic"] = d_list
            parsed = True

    cols = {c.lower(): c for c in df.columns}

    name_keys = ["nama","name","pasien","patient","user","id"]
    sys_keys  = ["sys","systolic","sistolik","sistole","atas","upper"]
    dia_keys  = ["dia","diastolic","diastolik","bawah","lower"]

    # Nama
    name_col = None
    for k in name_keys:
        for low, orig in cols.items():
            if k in low:
                name_col = orig
                break
        if name_col:
            break

    # Systolic
    sys_col = None
    for k in sys_keys:
        for low, orig in cols.items():
            if k in low:
                sys_col = orig
                break
        if sys_col:
            break

    # Diastolic
    dia_col = None
    for k in dia_keys:
        for low, orig in cols.items():
            if k in low:
                dia_col = orig
                break
        if dia_col:
            break

    # Kolom hasil parsing gabungan
    if not sys_col:
        cand = [c for c in df.columns if c.lower().endswith("_systolic")]
        if cand:
            sys_col = cand[0]
    if not dia_col:
        cand = [c for c in df.columns if c.lower().endswith("_diastolic")]
        if cand:
            dia_col = cand[0]

    # Jika masih belum ketemu → ambil kolom numerik
    if not sys_col or not dia_col:
        def is_num(s):
            try:
                pd.to_numeric(s.dropna().head(5))
                return True
            except:
                return False

        nums = [c for c in df.columns if is_num(df[c])]

        if len(nums) < 2:
            raise ValueError("Tidak cukup kolom tensi ditemukan.")

        if not sys_col:
            sys_col = nums[0]
        if not dia_col:
            dia_col = nums[1]

    # Jika tidak ada nama → buat
    if not name_col:
        df["Nama"] = [f"Pasien {i+1}" for i in range(len(df))]
        name_col = "Nama"

    df[sys_col] = pd.to_numeric(df[sys_col], errors="coerce")
    df[dia_col] = pd.to_numeric(df[dia_col], errors="coerce")

    df = df.dropna(subset=[sys_col, dia_col], how="all")

    return df.rename(columns={
        name_col: "Nama",
        sys_col: "Systolic",
        dia_col: "Diastolic"
    })
# ============================================================
# DETEKSI ANOMALI
# ============================================================

def detect_anomaly_df(df,
    thresh_sys_high=140,
    thresh_dia_high=90,
    thresh_sys_low=90,
    thresh_dia_low=60
):
    df = df.copy()
    df["Hipertensi"] = (df["Systolic"] > thresh_sys_high) | (df["Diastolic"] > thresh_dia_high)
    df["Hipotensi"]  = (df["Systolic"] < thresh_sys_low)  | (df["Diastolic"] < thresh_dia_low)
    df["Anom_Total"] = df["Hipertensi"] | df["Hipotensi"]
    return df

# ============================================================
# POP-UP NORMAL (HIJAU) — aman f-string
# ============================================================

def render_normal_overlay(duration_ms=2000):
    wav = generate_ting_wav(0.45)
    datauri = wav_bytes_to_datauri(wav)

    html = f"""
<div id="normal-root" style="
    position:fixed; inset:0; z-index:999999;
    background:rgba(0,0,0,0.45); backdrop-filter:blur(8px);
    display:flex; justify-content:center; align-items:center;
">

  <div style="
        width:500px; padding:32px;
        background:linear-gradient(135deg,#00d68f,#00aa6a);
        border-radius:28px; text-align:center; color:white;
        box-shadow:0 28px 60px rgba(0,80,40,0.45);
        position:relative; animation:popGreen .5s ease;
  ">

    <div onclick="document.getElementById('normal-root').remove();"
         style="
         position:absolute; top:10px; right:15px;
         background:rgba(255,255,255,0.25);
         border-radius:50%; width:34px; height:34px;
         display:flex; justify-content:center; align-items:center;
         cursor:pointer; font-size:18px; font-weight:700;">✖</div>

    <div style="
         font-size:88px;
         filter:drop-shadow(0 0 20px rgba(0,255,160,0.95));
         animation:glowGreen 1.3s infinite alternate;
    ">✔️</div>

    <h1 style="margin:0; font-size:38px; font-weight:900;">
        Datamu Normal!
    </h1>

    <p style="font-size:18px; opacity:0.95; margin-top:6px;">
        Jaga kesehatanmu selalu! 💚✨
    </p>

  </div>

  <audio autoplay>
    <source src="{datauri}" type="audio/wav">
  </audio>

</div>

<style>
@keyframes popGreen {{
    0% {{ transform:scale(.4); opacity:0; }}
    60% {{ transform:scale(1.12); opacity:1; }}
    100% {{ transform:scale(1); opacity:1; }}
}}

@keyframes glowGreen {{
    from {{ filter:drop-shadow(0 0 12px rgba(0,255,160,0.65)); }}
    to   {{ filter:drop-shadow(0 0 28px rgba(0,255,160,1)); }}
}}
</style>

<script>
setTimeout(function(){{
    var el = document.getElementById('normal-root');
    if(el) el.remove();
}}, {duration_ms});
</script>
"""
    st.markdown(html, unsafe_allow_html=True)

# ============================================================
# POP-UP WARNING (MERAH) — aman f-string
# ============================================================

def render_warning_inline(duration_ms=2200):
    wav = generate_siren_wav(1.0)
    datauri = wav_bytes_to_datauri(wav)

    html = f"""
<div id="warn-root" style="
    position:fixed; inset:0; z-index:999999;
    background:rgba(0,0,0,0.88); backdrop-filter:blur(12px);
    display:flex; justify-content:center; align-items:center;
">

  <div style="
        width:620px; background:linear-gradient(135deg,#ff2d2d,#8b0000);
        border-radius:30px; padding:40px 32px; text-align:center;
        color:white; box-shadow:0 40px 120px rgba(255,0,0,0.45);
        position:relative; animation:popWarn .6s ease;
  ">

    <div onclick="document.getElementById('warn-root').remove();" 
         style="
         position:absolute; top:10px; right:16px;
         background:rgba(255,255,255,0.22);
         border-radius:50%; width:34px; height:34px;
         display:flex; justify-content:center; align-items:center;
         font-size:20px; cursor:pointer; font-weight:700;">✖</div>

    <div style="
         font-size:108px;
         filter:drop-shadow(0 0 32px rgba(255,0,0,1));
         animation:glowWarn .5s infinite alternate;
    ">🚨</div>

    <h1 style="margin:0; font-size:42px; font-weight:900;">
        PERINGATAN TENSI TIDAK NORMAL!
    </h1>

    <p style="font-size:20px; margin-top:10px;">
        Hipertensi / Hipotensi terdeteksi — silakan cek ulang.
    </p>

  </div>

  <audio autoplay>
    <source src="{datauri}" type="audio/wav">
  </audio>

</div>

<style>
@keyframes popWarn {{
    0%   {{ transform:scale(.4); opacity:0; }}
    60%  {{ transform:scale(1.12); opacity:1; }}
    100% {{ transform:scale(1); opacity:1; }}
}}

@keyframes glowWarn {{
    from {{ filter:drop-shadow(0 0 20px rgba(255,90,90,0.85)); }}
    to   {{ filter:drop-shadow(0 0 42px rgba(255,0,0,1)); }}
}}
</style>

<script>
setTimeout(function(){{
    var el = document.getElementById('warn-root');
    if(el) el.remove();
}}, {duration_ms});
</script>
"""
    st.markdown(html, unsafe_allow_html=True)

# ============================================================
# BERANDA / LANDING PAGE
# ============================================================

if st.session_state.page == "beranda":

     st.markdown("<div class='big-nadi-title'>❤️ NADI : Numeric Analysis of Diastolic & Systolic</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='nadi-desc'><b>Adalah ruang sederhana untuk membaca alur tekanan darah Anda melalui pendekatan komputasi.</b><br>"
        "Dengan memanfaatkan metode <b>RK4</b> dan proses pengkodingan yang turut terbantu oleh kecerdasan buatan, <b>NADI</b> menghadirkan analisis yang ringan, intuitif, dan mudah dipahami.<br><br>"
        "<b>NADI bukan alat diagnosis medis</b>. Hasil yang ditampilkan hanya gambaran komputasi, bukan pengganti konsultasi tenaga kesehatan profesional.<br><br><i>Selamat datang. Biarkan NADI membaca aliran kesehatan Anda.</i></div>",
        unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="glass">
            <h3 style="color:#0b63d9;">Input Data Populasi</h3>
            <p style="color:#6b7280;">Upload CSV/XLSX, auto-detect tensi, prediksi RK4.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("➡ Masuk ke Input Data"):
            st.session_state.page = "input"

    with col2:
        st.markdown("""
        <div class="glass">
            <h3 style="color:#0b63d9;">Analisis Personal</h3>
            <p style="color:#6b7280;">Masukkan data tensi pribadi (1–10 titik).</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("➡ Masuk ke Personal"):
            st.session_state.page = "personal"

    st.markdown("---")

    stats = read_stats()
    st.markdown(f"""
    <div style='padding:14px; background:white; border-radius:14px;
                 box-shadow:0 6px 18px rgba(0,0,0,0.06); text-align:center;'>
        <h4 style="margin:0; color:#0b63d9;">📊 Statistik Penggunaan</h4><br>
        <b>Pengunjung Unik:</b> {stats["visitors"]}<br>
        <b>Total Analisis:</b> {stats["analyses"]}
    </div>
    """, unsafe_allow_html=True)

    st.stop()
# ============================================================
# INPUT DATA (UPLOAD CSV/XLSX)
# ============================================================

if st.session_state.page == "input":

    st.header("📁 Analisis Data Populasi (Upload CSV / XLSX)")

    uploaded = st.file_uploader(
        "Upload file CSV/XLSX (auto-detect tensi & nama)",
        type=["csv", "xlsx"]
    )

    run = st.button("Analisis (RK4)")

    if uploaded is not None:
        # ------------------------------
        # Load file
        # ------------------------------
        try:
            if uploaded.name.lower().endswith(".csv"):
                df_raw = pd.read_csv(uploaded)
            else:
                df_raw = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            st.stop()

        st.info("Preview 20 baris pertama:")
        st.dataframe(df_raw.head(20))

        # ------------------------------
        # AUTO-DETECT & CLEANING
        # ------------------------------
        try:
            df = autodetect_columns(df_raw)
        except Exception as e:
            st.error(f"Gagal mendeteksi kolom tensi: {e}")
            st.stop()

        # ------------------------------
        # Kolom Tanggal
        # ------------------------------
        if "Tanggal" in df.columns:
            df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
            df["Tanggal"] = df["Tanggal"].fillna(method="ffill")
        else:
            today = datetime.now().date()
            N = len(df)
            df["Tanggal"] = [pd.Timestamp(today - timedelta(days=(N-1-i))) for i in range(N)]

        # ======================================================
        # RUN ANALYSIS
        # ======================================================
        if run:

            all_parts = []
            alert_needed = False
            alert_list = []

            for name, g in df.groupby("Nama", sort=False):
                g2 = g.sort_values("Tanggal").reset_index(drop=True)
                g2 = detect_anomaly_df(g2)

                pred_s = rk4_predict_series(g2["Systolic"])
                pred_d = rk4_predict_series(g2["Diastolic"])

                g2["Prediksi_Systolic"] = np.nan
                g2["Prediksi_Diastolic"] = np.nan

                if pred_s is not None:
                    g2.at[len(g2)-1, "Prediksi_Systolic"] = pred_s
                    g2.at[len(g2)-1, "Prediksi_Diastolic"]  = pred_d

                all_parts.append(g2)

                if g2["Anom_Total"].any():
                    alert_needed = True
                    alert_list.append(name)

            result = pd.concat(all_parts, ignore_index=True)
            
            st.subheader("Hasil Analisis")
            st.dataframe(result)

            # ======================================================
            # Simpan hasil untuk halaman "Hasil Terakhir"
            # ======================================================
            st.session_state.last_result = result
            st.session_state.last_context = {
                "mode": "Input",
                "file": uploaded.name
            }

            # ======================================================
            # Update statistik
            # ======================================================
            stats = read_stats()
            stats["analyses"] += 1
            write_stats(stats)

            # ======================================================
            # DOWNLOAD BUTTONS
            # ======================================================
            csv_bytes = result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download hasil (CSV)", 
                csv_bytes,
                f"hasil_nadi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )

            try:
                out = BytesIO()
                with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
                    result.to_excel(writer, index=False, sheet_name="Hasil")
                    writer.save()
                out.seek(0)
                st.download_button(
                    "Download hasil (XLSX)", 
                    out,
                    f"hasil_nadi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except:
                pass

            # ======================================================
            # POPUP
            # ======================================================
            if alert_needed:
                st.error(f"🚨 Anomali terdeteksi pada: {', '.join(alert_list[:5])}")
                render_warning_inline()
            else:
                st.success("✔ Tidak ada tensi abnormal.")
                render_normal_overlay()

    if st.button("⬅ Kembali"):
        st.session_state.page = "beranda"

    st.stop()
# ============================================================
# PERSONAL ANALYSIS
# ============================================================

if st.session_state.page == "personal":
    st.header("👤 Analisis Personal")
    st.write("Isi hingga 10 data tensi lalu klik Analisis (RK4).")

    name = st.text_input("Nama")
    n = st.number_input("Jumlah data (1–10)", min_value=1, max_value=10, value=4)

    systolic = []
    diastolic = []

    for i in range(int(n)):
        c1, c2 = st.columns(2)
        with c1:
            s = st.number_input(f"Sistolik #{i+1}", min_value=40, max_value=250, value=120, key=f"s_{i}")
        with c2:
            d = st.number_input(f"Diastolik #{i+1}", min_value=20, max_value=180, value=80, key=f"d_{i}")
        systolic.append(s)
        diastolic.append(d)

    analyze = st.button("Analisis (RK4)")

    if analyze:
        if not name:
            st.error("Masukkan nama terlebih dahulu.")
            st.stop()

        dfp = pd.DataFrame({
            "Nama":[name]*int(n),
            "Tanggal":[pd.Timestamp(datetime.now().date() - timedelta(days=(int(n)-1-i))) for i in range(int(n))],
            "Systolic":systolic,
            "Diastolic":diastolic
        })

        dfp = detect_anomaly_df(dfp)

        pred_s = rk4_predict_series(dfp["Systolic"])
        pred_d = rk4_predict_series(dfp["Diastolic"])

        dfp["Prediksi_Systolic"] = np.nan
        dfp["Prediksi_Diastolic"] = np.nan

        if pred_s is not None:
            dfp.at[len(dfp)-1, "Prediksi_Systolic"] = pred_s
            dfp.at[len(dfp)-1, "Prediksi_Diastolic"] = pred_d

        st.subheader("Hasil Analisis Personal")
        st.dataframe(dfp)

        # Grafik tensi
        fig, ax = plt.subplots(figsize=(9,3))
        ax.plot(dfp["Tanggal"], dfp["Systolic"], marker="o", label="Systolic")
        ax.plot(dfp["Tanggal"], dfp["Diastolic"], marker="o", label="Diastolic")

        if pred_s is not None:
            nd = dfp["Tanggal"].iloc[-1] + pd.Timedelta(days=1)
            ax.scatter([nd],[pred_s], marker='D', s=80)
            ax.scatter([nd],[pred_d], marker='D', s=80)

        ax.set_title(f"Tensi - {name}")
        ax.legend()
        st.pyplot(fig)

        # anomaly check
        if dfp["Anom_Total"].iloc[-1]:
            st.error("⚠️ Terdeteksi hipertensi / hipotensi!")
            render_warning_inline()
        else:
            render_normal_overlay()
            st.success("✔ Datamu Normal. Jaga kesehatan yaaa!!")

        if pred_s is not None:
            st.markdown(f"**Prediksi RK4** → Sistolik: **{pred_s:.2f}**, Diastolik: **{pred_d:.2f}**")

        # simpan hasil
        st.session_state.last_result = dfp
        st.session_state.last_context = {"mode":"Personal", "name":name}

        # update statistik
        stats = read_stats()
        stats["analyses"] += 1
        write_stats(stats)

    if st.button("⬅ Kembali"):
        st.session_state.page = "beranda"

    st.stop()

# ============================================================
# RESULTS (LAST)
# ============================================================

if st.session_state.page == "hasil":
    st.header("📊 Hasil Analisis Terakhir")

    if st.session_state.last_result is None:
        st.info("Belum ada hasil. Lakukan analisis pada menu Input Data atau Personal.")
    else:
        st.write("Context:", st.session_state.last_context)
        st.dataframe(st.session_state.last_result)

        df_show = st.session_state.last_result

        if "Anom_Total" in df_show.columns:
            total_anom = int(df_show["Anom_Total"].sum())
            st.markdown(f"**Total hipertensi/hipotensi terdeteksi: {total_anom}**")

        # additional quick metrics
        if "Prediksi_Systolic" in df_show.columns:
            preds = df_show[["Nama","Prediksi_Systolic","Prediksi_Diastolic"]].dropna(how="all")
            if not preds.empty:
                st.markdown("**Prediksi berikutnya (ringkasan):**")
                st.dataframe(preds.head(10))

    if st.button("⬅ Kembali"):
        st.session_state.page = "beranda"

    st.stop()

# ============================================================
# WHY RK4 PAGE
# ============================================================

if st.session_state.page == "rk4info":
    st.header("❔ Mengapa Menggunakan Metode RK4?")

    st.markdown("""
    Metode **Runge–Kutta Orde 4 (RK4)** adalah salah satu pendekatan numerik yang paling
    banyak digunakan ketika kita ingin memperkirakan nilai di masa depan berdasarkan data
    yang sudah ada. Dalam konteks aplikasi ini, RK4 dipakai untuk memperkirakan arah
    perubahan tekanan darah sehingga pengguna dapat melihat pola yang lebih halus dan
    stabil. Ketika data tekanan darah berubah-ubah karena aktivitas, kondisi tubuh, atau
    sedikit kesalahan saat pengukuran, hasilnya sering terlihat kaku atau mendadak.
    RK4 membantu meredam efek tersebut dengan cara melakukan proses perhitungan secara
    bertahap dan lebih hati-hati, sehingga hasil akhirnya menjadi lebih konsisten.

    Sederhananya, RK4 bekerja dengan membaca perubahan secara perlahan dari beberapa
    sudut pandang perhitungan, lalu menggabungkannya menjadi satu prediksi yang lebih
    masuk akal. Inilah mengapa banyak sistem monitoring memakai RK4: metodenya cukup
    akurat, namun tidak memberatkan komputasi. Jika dibandingkan dengan metode yang
    sangat sederhana, hasil RK4 jauh lebih stabil dan tidak mudah terpengaruh oleh satu
    atau dua data yang tiba-tiba saja naik atau turun drastis. Hasil prediksi tidak
    melompat-lompat, dan lebih mudah dianalisis.

    Dalam aplikasi ini, RK4 tidak dimaksudkan untuk diagnosis, melainkan sebagai alat
    edukasi guna membantu melihat pola perubahan tekanan darah dengan cara yang lebih
    halus dan aman. Karena sifatnya yang stabil dan tidak terlalu sensitif terhadap
    fluktuasi kecil, RK4 menjadi pilihan terbaik untuk menampilkan gambaran umum pola
    tekanan darah tanpa membuat pengguna salah paham akibat perubahan-perubahan kecil
    yang sebenarnya tidak signifikan.
    """)

    if st.button("⬅ Kembali"):
        st.session_state.page = "beranda"

    st.stop()

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    "<div style='color:#6b7280; font-size:13px;'>"
    "NADI | RK4 — aplikasi edukasi numerik, bukan pengganti konsultasi medis."
    "</div>",
    unsafe_allow_html=True
)
