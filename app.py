# app.py — NADI (RK4) — FINAL (Warning super-dramatic inline + Normal gemoy overlay)
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

# -----------------------
# Page config & session
# -----------------------
st.set_page_config(page_title="NADI (RK4) — Soft Blue", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "beranda"
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_context" not in st.session_state:
    st.session_state.last_context = None

# -----------------------
# GLOBAL CSS (app look)
# -----------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    :root{ --bg1: #f3f9ff; --bg2: #eaf6ff; --primary-1: #0b63d9; --muted: #6b7280; }
    html, body, [class*="css"] { font-family: Inter, "Times New Roman", serif; background: linear-gradient(180deg, var(--bg1), var(--bg2)); }
    .big-nadi-title { font-family: "Times New Roman", serif; font-size:70px; font-weight:800; color:white !important; text-shadow:0 0 14px rgba(0,0,0,0.45); text-align:center; margin-top:4px; margin-bottom:8px; }
    .nadi-desc { text-align:center; color:var(--muted); max-width:920px; margin:auto; margin-bottom:18px; font-size:16px; line-height:1.45; }
    .glass { background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(246,251,255,0.88)); border-radius:14px; padding:18px; box-shadow:0 8px 28px rgba(11,99,217,0.06); border:1px solid rgba(255,255,255,0.6); transition: transform .18s ease, box-shadow .18s ease; }
    .glass:hover { transform: translateY(-6px); box-shadow:0 18px 40px rgba(11,99,217,0.09); }
    .spacer { margin-top:14px; margin-bottom:14px; }
    @media (max-width:600px) { .big-nadi-title { font-size:36px; } }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    button.bigglass {
    background: rgba(255, 255, 255, 0.55) !important;
    backdrop-filter: blur(6px);
    border: 1px solid rgba(255,255,255,0.7) !important;
    padding: 18px 32px !important;
    font-size: 20px !important;
    font-weight: 600 !important;
    border-radius: 16px !important;
    color: #0b63d9 !important;
    box-shadow: 0 4px 18px rgba(11,99,217,0.18);
    transition: all 0.2s ease-in-out;
    }

    button.bigglass:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 28px rgba(11,99,217,0.25);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------
# AUDIO HELPERS (siren + ting)
# -----------------------
def generate_siren_wav(duration=6.0, sr=44100):
    # siren with pitch modulation (short)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    mod = 0.5 * (1 + np.sin(2 * np.pi * 2.2 * t))  # faster mod for urgency
    freq = 700 + 600 * np.sin(2 * np.pi * 1.0 * t)  # oscillating pitch
    tone = 0.9 * np.sin(2 * np.pi * freq * t) * (0.6 + 0.4 * mod)
    # apply short fade-out to avoid clicks
    env = np.linspace(1, 0.01, len(t))
    tone = tone * env
    tone = np.clip(tone, -1, 1)
    buf = BytesIO()
    sf.write(buf, tone, sr, format='WAV')
    buf.seek(0)
    return buf.read()

def generate_ting_wav(duration=3.0, sr=44100):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    tone1 = 0.7 * np.sin(2 * np.pi * 1400 * t) * np.linspace(1, 0, len(t))
    tone2 = 0.5 * np.sin(2 * np.pi * 1800 * t) * np.linspace(1, 0, len(t))
    tone = tone1 + 0.6 * tone2
    tone = np.clip(tone, -1, 1)
    buf = BytesIO()
    sf.write(buf, tone, sr, format='WAV')
    buf.seek(0)
    return buf.read()

def wav_bytes_to_datauri(wav_bytes):
    b64 = base64.b64encode(wav_bytes).decode()
    return f"data:audio/wav;base64,{b64}"

# -----------------------
# RK4 PREDICTION
# -----------------------
def rk4_predict_value(last, prev, h=1.0):
    slope = last - prev
    def f(t, y): return slope
    k1 = f(0, last)
    k2 = f(h/2, last + h*k1/2)
    k3 = f(h/2, last + h*k2/2)
    k4 = f(h, last + h*k3)
    return last + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

def rk4_predict_series(arr):
    arr = list(arr)
    if len(arr) < 2:
        return None
    return rk4_predict_value(arr[-1], arr[-2])

# -----------------------
# ANOMALY DETECTION
# -----------------------
def detect_anomaly_df(df,
                      thresh_sys_high=140,
                      thresh_dia_high=90,
                      thresh_sys_low=90,
                      thresh_dia_low=60):
    df = df.copy()
    df['Systolic'] = pd.to_numeric(df['Systolic'], errors='coerce')
    df['Diastolic'] = pd.to_numeric(df['Diastolic'], errors='coerce')
    df['Hipertensi'] = (df['Systolic'] > thresh_sys_high) | (df['Diastolic'] > thresh_dia_high)
    df['Hipotensi'] = (df['Systolic'] < thresh_sys_low) | (df['Diastolic'] < thresh_dia_low)
    df['Anom_Total'] = df['Hipertensi'] | df['Hipotensi']
    return df

# -----------------------
# RENDER NORMAL — INLINE FULLSCREEN (overlay via st.markdown)
# -----------------------
def render_normal_overlay(datauri=None, duration_ms=1500):
    import time
    uid = str(int(time.time() * 1000))

    audio_html = ""
    if datauri:
        audio_html = f'<audio autoplay><source src="{datauri}" type="audio/wav"></audio>'

    html = f"""
    <div id="normal-root-{uid}" class="nadi-popup-normal">
        <div class="nadi-popup-box">
            <div style="font-size:90px; margin-bottom:10px;">✔️</div>
            <h2 style="margin:0; font-size:34px; font-weight:900;">Datamu Normal!</h2>
            <p style="font-size:20px; opacity:0.95; margin-top:8px;">Jaga kesehatan yaaa!! 💚✨</p>
        </div>
        {audio_html}
    </div>

    <style>
    /* Overlay background */
    #normal-root-{uid} {{
        position:fixed; inset:0;
        background:rgba(0,0,0,0.55);
        backdrop-filter:blur(7px);
        z-index:999999;
        display:flex; justify-content:center; align-items:center;
        animation: zoomIn 0.35s ease forwards;
    }}

    /* Box */
    #normal-root-{uid} .nadi-popup-box {{
        background:linear-gradient(135deg,#00e09f,#00b46f);
        border-radius:30px;
        padding:32px;
        text-align:center;
        color:white;
        box-shadow:0 40px 80px rgba(0,50,20,0.4);
        animation: zoomIn 0.35s ease forwards;
    }}

    /* ZOOM IN */
    @keyframes zoomIn {{
        0% {{ transform:scale(0.4); opacity:0; }}
        80% {{ transform:scale(1.15); opacity:1; }}
        100% {{ transform:scale(1); opacity:1; }}
    }}

    /* ZOOM OUT + FADE OUT */
    @keyframes zoomFadeOut {{
        0% {{ transform:scale(1); opacity:1; }}
        100% {{ transform:scale(0.7); opacity:0; }}
    }}
    </style>

    <script>
        // jalankan fade-out setelah durasi
        setTimeout(function() {{
            var el = document.getElementById("normal-root-{uid}");
            if (el) {{
                el.style.animation = "zoomFadeOut 0.6s ease forwards";
                setTimeout(function() {{
                    if (el && el.parentNode) el.remove();
                }}, 600);
            }}
        }}, {duration_ms});
    </script>
    """

    st.markdown(html, unsafe_allow_html=True)

# -----------------------
# RENDER WARNING — INLINE FULLSCREEN (super dramatic) + siren (1s)
# -----------------------
def render_warning_inline(duration_ms=1200):
    import time
    uid = str(int(time.time() * 1000))
    wav = generate_siren_wav(duration=1.0)
    datauri = wav_bytes_to_datauri(wav)

    html = f"""
    <div id="warn-root-{uid}" class="nadi-popup-warn">
        <div class="warn-box">
            <div style="font-size:100px; margin-bottom:10px;">🚨</div>
            <h1 style="margin:0; font-size:36px; font-weight:900;">PERINGATAN TENSI TIDAK NORMAL!</h1>
            <p style="font-size:20px; opacity:0.95;">Hipertensi / hipotensi terdeteksi.</p>
        </div>

        <audio autoplay>
            <source src="{datauri}" type="audio/wav">
        </audio>
    </div>

    <style>
    #warn-root-{uid} {{
        position:fixed; inset:0;
        background:rgba(0,0,0,0.88);
        backdrop-filter:blur(10px);
        z-index:999999;
        display:flex; justify-content:center; align-items:center;
        animation: zoomIn 0.45s ease forwards;
    }}

    #warn-root-{uid} .warn-box {{
        background:linear-gradient(135deg,#ff2d2d,#8b0000);
        padding:40px 32px;
        border-radius:30px;
        text-align:center;
        color:white;
        box-shadow:0 40px 120px rgba(255,0,0,0.45);
        animation: zoomIn 0.45s ease forwards;
    }}

    /* zoom in */
    @keyframes zoomIn {{
        0% {{ transform:scale(0.3); opacity:0; }}
        75% {{ transform:scale(1.18); opacity:1; }}
        100% {{ transform:scale(1); opacity:1; }}
    }}

    /* zoom out + fade out */
    @keyframes zoomFadeOut {{
        0% {{ transform:scale(1); opacity:1; }}
        100% {{ transform:scale(0.6); opacity:0; }}
    }}
    </style>

    <script>
        setTimeout(function() {{
            var el = document.getElementById("warn-root-{uid}");
            if (el) {{
                el.style.animation = "zoomFadeOut 0.65s ease forwards";
                setTimeout(function() {{
                    if (el && el.parentNode) el.remove();
                }}, 650);
            }}
        }}, {duration_ms});
    </script>
    """

    st.markdown(html, unsafe_allow_html=True)

# ============================================================
# BERANDA / LANDING
# ============================================================
if st.session_state.page == "beranda":
    st.markdown("<div class='big-nadi-title'>❤️ NADI : Numeric Analysis of Diastolic & Systolic</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='nadi-desc'><b>Adalah ruang sederhana untuk membaca alur tekanan darah Anda melalui pendekatan komputasi.</b><br>"
        "Dengan memanfaatkan metode <b>RK4</b> dan proses pengkodingan yang turut terbantu oleh kecerdasan buatan, <b>NADI</b> menghadirkan analisis yang ringan, intuitif, dan mudah dipahami.<br><br>"
        "<b>NADI bukan alat diagnosis medis</b>. Hasil yang ditampilkan hanya gambaran komputasi, bukan pengganti konsultasi tenaga kesehatan profesional.<br><br><i>Selamat datang. Biarkan NADI membaca aliran kesehatan Anda.</i></div>",
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='glass'><h3 style='color:var(--primary-1)'>Input Data (Puskesmas / Klinik)</h3>"
                    "<p style='color:var(--muted)'>Upload CSV/XLSX berisi banyak pasien. Sistem akan memproses tiap pasien, mendeteksi hipertensi/hipotensi, dan memprediksi nilai berikutnya.</p>"
                    "<div class='spacer'></div>", unsafe_allow_html=True)
        if st.button("➡ Masuk ke Input Data", key="go_input"):
            st.session_state.page = "input"
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='glass'><h3 style='color:var(--primary-1)'>Analisis Personal</h3>"
                    "<p style='color:var(--muted)'>Masukkan 1–10 data tensi untuk analisis dan prediksi personal.</p>"
                    "<div class='spacer'></div>", unsafe_allow_html=True)
        if st.button("➡ Masuk ke Personal", key="go_personal"):
            st.session_state.page = "personal"
        st.markdown("</div>", unsafe_allow_html=True)

    a, b, c = st.columns(3)
    with a:
        st.markdown('<button class="bigglass" onclick="document.querySelector(\'button[kind=primary]\').click()">📊 Hasil Analisis Terakhir</button>', unsafe_allow_html=True)
        if st.button("📊 Hasil Analisis Terakhir", key="btn_hasil", help="trigger"):
            st.session_state.page = "hasil"
    with b:
        st.markdown('<button class="bigglass" onclick="document.querySelector(\'button[kind=secondary]\').click()">❔ Mengapa RK4?</button>', unsafe_allow_html=True)
        if st.button("❔ Mengapa RK4?", key="btn_rk4", help="trigger"):
             st.session_state.page = "rk4info"
    with c:
        st.markdown('<button class="bigglass" onclick="document.querySelector(\'button[kind=tertiary]\').click()">🔄 Reset Hasil</button>', unsafe_allow_html=True)
        if st.button("🔄 Reset Hasil", key="btn_reset", help="trigger"):
             st.session_state.last_result = None
             st.session_state.last_context = None
             st.success("Riwayat berhasil dibersihkan.")


    st.markdown("---")
    st.subheader("Contoh Template Data (Upload)")
    sample_df = pd.DataFrame({
        "Nama": ["Budi","Budi","Siti","Siti"],
        "Tanggal": [
            datetime.now().date()-timedelta(days=3),
            datetime.now().date()-timedelta(days=1),
            datetime.now().date()-timedelta(days=2),
            datetime.now().date()
        ],
        "Systolic": [120, 145, 130, 170],
        "Diastolic": [80,  95,  85, 105]
    })
    st.dataframe(sample_df)
    csv = sample_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download template CSV", csv, "template_tensi.csv", "text/csv")
    st.stop()

# ============================================================
# INPUT DATA (UPLOAD) - REVISI (menggunakan st.form + debug)
# ============================================================
if st.session_state.page == "input":
    st.header("📁 Analisis Data Populasi (Upload CSV / XLSX)")

    # Form: file_uploader + submit dalam satu interaksi
    with st.form("upload_form", clear_on_submit=False):
        uploaded = st.file_uploader("Upload CSV / XLSX (minimal kolom: Nama, Systolic, Diastolic)", type=["csv","xlsx"])
        submitted = st.form_submit_button("Analisis (RK4)")

    # Quick debug info tentang file
    if uploaded is not None:
        st.info(f"File terdeteksi: **{uploaded.name}** — ukuran: {getattr(uploaded, 'size', 'n/a')} bytes")
        try:
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            st.stop()

        # Bersihkan nama kolom
        df.columns = [c.strip() for c in df.columns]
        st.info(f"Kolom terdeteksi: {list(df.columns)}")
        st.dataframe(df.head(30))

        required = {"Nama","Systolic","Diastolic"}
        if not required.issubset(df.columns):
            st.error(f"Kolom minimal harus ada: {sorted(required)}. Periksa header CSV (spasi / BOM / encoding).")
            st.stop()

        # handle tanggal
        if "Tanggal" in df.columns:
            df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
            if df["Tanggal"].isna().any():
                df["Tanggal"] = df["Tanggal"].fillna(method="ffill")
        else:
            today = datetime.now().date()
            N = len(df)
            df["Tanggal"] = [pd.Timestamp(today - timedelta(days=(N-1-i))) for i in range(N)]

        # Jika user menekan Analisis
        if submitted:
            try:
                parts = []
                alert_needed = False
                alert_names = []
                processing_errors = []

                # Pastikan kolom numeric
                df["Systolic"] = pd.to_numeric(df["Systolic"], errors="coerce")
                df["Diastolic"] = pd.to_numeric(df["Diastolic"], errors="coerce")

                # Tampilkan ringkasan sebelum loop
                st.write(f"Memulai analisis untuk {df['Nama'].nunique()} pasien, total baris: {len(df)}")

                for name, g in df.groupby("Nama", sort=False):
                    try:
                        g2 = g.sort_values("Tanggal").reset_index(drop=True)
                        g2 = g2[["Nama","Tanggal","Systolic","Diastolic"]].copy()

                        # Jika semua nilai NaN untuk tensi -> lewati & catat
                        if g2["Systolic"].isna().all() and g2["Diastolic"].isna().all():
                            processing_errors.append(f"{name}: semua nilai Systolic/Diastolic kosong. Dilewati.")
                            continue

                        g2 = detect_anomaly_df(g2)

                        pred_s = rk4_predict_series(g2["Systolic"])
                        pred_d = rk4_predict_series(g2["Diastolic"])

                        g2["Prediksi_Systolic"] = np.nan
                        g2["Prediksi_Diastolic"] = np.nan
                        if pred_s is not None:
                            g2.at[len(g2)-1, "Prediksi_Systolic"] = pred_s
                            g2.at[len(g2)-1, "Prediksi_Diastolic"] = pred_d

                        parts.append(g2)
                        if g2["Anom_Total"].any():
                            alert_needed = True
                            alert_names.append(name)
                    except Exception as e:
                        processing_errors.append(f"{name}: error saat proses -> {e}")

                if len(parts) == 0:
                    st.warning("Tidak ada data pasien yang berhasil diproses. Periksa isi file (baris/kolom/format).")
                else:
                    result = pd.concat(parts, ignore_index=True)
                    st.subheader("Hasil Analisis")
                    st.dataframe(result)
                    st.session_state.last_result = result
                    st.session_state.last_context = {"mode":"Input","file":uploaded.name}

                    if alert_needed:
                        st.error(f"🚨 Anomali terdeteksi pada: {', '.join(alert_names[:8])}")
                        render_warning_inline(duration_ms=1000)
                    else:
                        wav = generate_ting_wav(duration=0.45)
                        datauri = wav_bytes_to_datauri(wav)
                        render_normal_overlay(datauri=datauri, duration_ms=1400)
                        st.success("✔ Tidak ada hipertensi/hipotensi terdeteksi.")

                # tampilkan log error per-pasien bila ada
                if processing_errors:
                    st.markdown("**Catatan pemrosesan (beberapa entry dilewati / error):**")
                    for msg in processing_errors:
                        st.markdown(f"- {msg}")
                
                st.markdown("---")
                if st.button("⬅ Kembali ke Beranda", key="back_from_personal_result"):
                    st.session_state.page = "beranda"
                    st.rerun()

            except Exception as e:
                st.error(f"Terjadi error saat analisis: {e}")
                import traceback
                st.text(traceback.format_exc())

    # Back button
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
            "Tanggal":[pd.Timestamp(datetime.now().date()-timedelta(days=(int(n)-1-i))) for i in range(int(n))],
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

        # Chart
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

        # anomaly handling
        if dfp["Anom_Total"].iloc[-1]:
            st.error("⚠️ Terdeteksi hipertensi / hipotensi!")
            render_warning_inline(duration_ms=1000)
        else:
            wav = generate_ting_wav(duration=0.45)
            datauri = wav_bytes_to_datauri(wav)
            render_normal_overlay(datauri=datauri, duration_ms=1400)
            st.success("✔ Datamu Normal. Jaga Kesehatan Yaa!!!")

        if pred_s is not None:
            st.markdown(f"**Prediksi RK4 (1 langkah)** — Sistolik: **{pred_s:.2f}**, Diastolik: **{pred_d:.2f}**")

        st.session_state.last_result = dfp
        st.session_state.last_context = {"mode":"Personal", "name":name}

        st.markdown("---")
        if st.button("⬅ Kembali ke Beranda", key="back_from_personal_result"):
             st.session_state.page = "beranda"
             st.rerun()

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
        total_anom = int(df_show["Anom_Total"].sum()) if "Anom_Total" in df_show.columns else 0
        st.markdown(f"**Total hipertensi / hipotensi terdeteksi:** {total_anom}")

    if st.button("⬅ Kembali"):
        st.session_state.page = "beranda"
    st.stop()

# ============================================================
# WHY RK4
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
st.markdown("<div style='color:var(--muted); font-size:13px;'>NADI | RK4 — aplikasi edukasi numerik, bukan alat diagnosis medis.</div>", unsafe_allow_html=True)
