# ============================================================
# ===================== IMPORT LIBRARY =======================
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import time
import base64
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import soundfile as sf
from io import BytesIO
import streamlit.components.v1 as components

# ============================================================
# ===================== PAGE CONFIG ===========================
# ============================================================

st.set_page_config(page_title="NADI (RK4) — Soft Blue", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "beranda"
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_context" not in st.session_state:
    st.session_state.last_context = None


# ============================================================
# ===================== CUSTOM STYLING ========================
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    :root{
      --bg1: #f3f9ff;
      --bg2: #eaf6ff;
      --primary-1: #0b63d9;
      --primary-2: #1ea7ff;
      --muted: #6b7280;
    }
    html, body, [class*="css"] {
        font-family: Inter, "Times New Roman", serif;
        background: linear-gradient(180deg, var(--bg1), var(--bg2));
    }
    .big-nadi-title {
        font-family: "Times New Roman", serif;
        font-size: 70px;
        font-weight: 800;
        color: white !important;
        text-shadow: 0 0 14px rgba(0,0,0,0.45);
        text-align: center;
        margin-top: 4px;
        margin-bottom: 8px;
    }
    .nadi-desc {
        text-align:center;
        color: var(--muted);
        max-width: 920px;
        margin:auto;
        margin-bottom:18px;
        font-size:16px;
        line-height:1.45;
    }
    .glass {
        background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(246,251,255,0.88));
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 8px 28px rgba(11,99,217,0.06);
        border: 1px solid rgba(255,255,255,0.6);
        transition: transform .18s ease, box-shadow .18s ease;
    }
    .glass:hover {
        transform: translateY(-6px);
        box-shadow: 0 18px 40px rgba(11,99,217,0.09);
    }
    .spacer { margin-top: 14px; margin-bottom: 14px; }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# ===================== AUDIO GENERATOR =======================
# ============================================================

def generate_ting_wav(duration=0.8, sr=44100):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq = 2000
    tone = 0.8 * np.sin(2 * np.pi * freq * t)
    tone *= np.linspace(1, 0, len(t))
    buffer = BytesIO()
    sf.write(buffer, tone, sr, format="WAV")
    buffer.seek(0)
    return buffer.read()

def generate_siren_wav(duration=0.8, sr=44100):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freq = 900 + 180 * np.sin(2 * np.pi * 2 * t)
    tone = 0.7 * np.sin(2 * np.pi * freq * t)
    tone *= np.linspace(1, 0, len(t))
    buffer = BytesIO()
    sf.write(buffer, tone, sr, format="WAV")
    buffer.seek(0)
    return buffer.read()

def wav_bytes_to_datauri(wav_bytes):
    b64 = base64.b64encode(wav_bytes).decode()
    return f"data:audio/wav;base64,{b64}"

# ============================================================
# ===================== RK4 & DETEKSI =========================
# ============================================================

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


def detect_anomaly_df(df):
    df = df.copy()
    df['Systolic'] = pd.to_numeric(df['Systolic'], errors='coerce')
    df['Diastolic'] = pd.to_numeric(df['Diastolic'], errors='coerce')
    df['Hipertensi'] = (df['Systolic'] > 140) | (df['Diastolic'] > 90)
    df['Hipotensi'] = (df['Systolic'] < 90) | (df['Diastolic'] < 60)
    df['Anom_Total'] = df['Hipertensi'] | df['Hipotensi']
    return df

# ============================================================
# ===================== POPUP DRAMATIC FIX ===================
# ============================================================

def warning_popup():
    html = """
    <style>
    .popup-root {
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0);
      backdrop-filter: blur(0px);
      animation: fadeBG 0.4s forwards;
      display:flex;
      align-items:center;
      justify-content:center;
      z-index:999999 !important;
    }

    @keyframes fadeBG {
      0% { background: rgba(0,0,0,0); backdrop-filter: blur(0px); }
      100% { background: rgba(0,0,0,0.75); backdrop-filter: blur(8px); }
    }

    .popup-box {
      width:520px;
      background:linear-gradient(135deg,#ff4444,#b00000);
      border-radius:22px;
      padding:40px;
      color:white;
      text-align:center;
      animation:popin .35s ease-out, shake 0.4s ease-in-out 0.35s;
      box-shadow:0 25px 55px rgba(0,0,0,0.5);
    }

    @keyframes popin {
      0% {transform:scale(.4); opacity:0;}
      60% {transform:scale(1.15); opacity:1;}
      100% {transform:scale(1); opacity:1;}
    }

    @keyframes shake {
      0% { transform: translateX(0); }
      25% { transform: translateX(-10px); }
      50% { transform: translateX(12px); }
      75% { transform: translateX(-8px); }
      100% { transform: translateX(0); }
    }

    .icon {
      font-size:90px;
      margin-bottom:10px;
      animation: glow 1s infinite alternate;
    }

    @keyframes glow {
      from { filter: drop-shadow(0 0 5px #ff9090); }
      to   { filter: drop-shadow(0 0 20px #ff1e1e); }
    }
    </style>

    <div class="popup-root" id="popup-root">
      <div class="popup-box">
        <div class="icon">🚨</div>
        <h2 style="font-size:32px; margin:0; font-weight:900;">TERDETEKSI ANOMALI</h2>
        <p style="font-size:20px; margin-top:6px;">Hipertensi / Hipotensi Terdeteksi.</p>
      </div>
    </div>

    <script>
    setTimeout(() => {
        let x = document.getElementById("popup-root");
        if (x) x.remove();
    }, 1900);
    </script>
    """
    components.html(html, height=0, width=0, scrolling=False)



def normal_popup():
    html = """
    <style>
    .green-root {
      position:fixed; inset:0;
      background:rgba(0,0,0,0);
      backdrop-filter:blur(0px);
      animation: fadeBG 0.4s forwards;
      display:flex;
      justify-content:center;
      align-items:center;
      z-index:999999 !important;
    }

    @keyframes fadeBG {
      0% { background: rgba(0,0,0,0); backdrop-filter: blur(0px); }
      100% { background: rgba(0,0,0,0.55); backdrop-filter: blur(6px); }
    }

    .green-box {
      background:linear-gradient(135deg,#00b35a,#38ff9f);
      padding:44px 50px;
      border-radius:22px;
      text-align:center;
      color:white;
      animation: bounceIn 0.5s ease-out;
      box-shadow:0 0 30px rgba(0,0,0,0.45);
      position: relative;
    }

    @keyframes bounceIn {
      0% { transform:scale(0.3); opacity:0; }
      60% { transform:scale(1.15); opacity:1; }
      100% { transform:scale(1); opacity:1; }
    }

    .green-icon {
      font-size:80px;
      margin-bottom:10px;
      animation: pulse 1.2s infinite;
    }

    @keyframes pulse {
      0% { transform:scale(1); }
      50% { transform:scale(1.12); }
      100% { transform:scale(1); }
    }

    /* sparkles */
    .sparkle {
      position:absolute;
      width:10px;
      height:10px;
      background:white;
      border-radius:50%;
      opacity:0;
      animation: sparkleAnim 1.6s infinite;
    }

    .sparkle:nth-child(1) { top:10px; left:20px; animation-delay:0s; }
    .sparkle:nth-child(2) { top:50px; right:18px; animation-delay:0.3s; }
    .sparkle:nth-child(3) { bottom:15px; left:40px; animation-delay:0.6s; }

    @keyframes sparkleAnim {
      0% { opacity:0; transform:scale(0.4); }
      50% { opacity:1; transform:scale(1.2); }
      100%{ opacity:0; transform:scale(0.4); }
    }
    </style>

    <div class="green-root" id="green-pop">
       <div class="green-box">
         <div class="green-icon">✔️</div>
         <h2 style="font-size:30px; margin:0; font-weight:900;">DATAMU NORMAL</h2>
         <p style="font-size:20px; opacity:0.95;">Jaga kesehatan ya!!! 💚✨</p>

         <div class="sparkle"></div>
         <div class="sparkle"></div>
         <div class="sparkle"></div>
       </div>
    </div>

    <script>
    setTimeout(() => {
        let x = document.getElementById("green-pop");
        if (x) x.remove();
    }, 2000);
    </script>
    """
    components.html(html, height=0, width=0, scrolling=False)

# ============================================================
# ====================   BERANDA / LANDING   =================
# ============================================================

if st.session_state.page == "beranda":

    st.markdown(
        "<div class='big-nadi-title'>❤️ NADI : Numeric Analysis of Diastolic & Systolic</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='nadi-desc'><b>Adalah ruang sederhana untuk membaca alur tekanan darah Anda melalui pendekatan komputasi.</b><br>Dengan memanfaatkan metode <b>RK4</b> dan bantuan kecerdasan buatan, <b>NADI</b> menghadirkan analisis yang ringan, intuitif, dan mudah dipahami.<br><br>Namun, <b>NADI bukan alat diagnosis medis</b>. Hasil yang ditampilkan hanya gambaran komputasi, bukan pengganti konsultasi tenaga kesehatan profesional.<br><br><i>Selamat datang. Biarkan NADI membaca aliran kesehatan Anda.</i></div>",
        unsafe_allow_html=True
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            "<div class='glass'><h3 style='color:var(--primary-1)'>Input Data (Puskesmas / Klinik)</h3>"
            "<p style='color:var(--muted)'>Upload CSV/XLSX berisi banyak pasien untuk analisis massal menggunakan RK4.</p>"
            "<div class='spacer'></div>",
            unsafe_allow_html=True
        )
        if st.button("➡ Masuk ke Input Data"):
            st.session_state.page = "input"
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(
            "<div class='glass'><h3 style='color:var(--primary-1)'>Analisis Personal</h3>"
            "<p style='color:var(--muted)'>Masukkan 1–10 data tensi untuk analisis dan prediksi personal.</p>"
            "<div class='spacer'></div>",
            unsafe_allow_html=True
        )
        if st.button("➡ Masuk ke Personal"):
            st.session_state.page = "personal"
        st.markdown("</div>", unsafe_allow_html=True)

    a,b,c = st.columns(3)
    with a:
        if st.button("📊 Hasil Analisis Terakhir"):
            st.session_state.page = "hasil"
    with b:
        if st.button("❔ Mengapa RK4?"):
            st.session_state.page = "rk4info"
    with c:
        if st.button("🔄 Reset Hasil"):
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
    st.download_button("Download template CSV", csv, "template_tensi.csv","text/csv")

    st.stop()


# ============================================================
# ==================   INPUT DATA (UPLOAD)   =================
# ============================================================

if st.session_state.page == "input":

    st.header("📁 Analisis Data Populasi (Upload CSV / XLSX)")
    uploaded = st.file_uploader("Upload CSV/XLSX", type=["csv","xlsx"])
    run = st.button("Analisis (RK4)")

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            st.stop()

        df.columns = [c.strip() for c in df.columns]

        required = {"Nama","Systolic","Diastolic"}
        if not required.issubset(df.columns):
            st.error(f"Kolom wajib hilang. Minimal harus ada {required}")
            st.stop()

        st.info("Pratinjau data:")
        st.dataframe(df.head(50))

        if "Tanggal" in df.columns:
            df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
            df["Tanggal"] = df["Tanggal"].fillna(method="ffill")
        else:
            today = datetime.now().date()
            N = len(df)
            df["Tanggal"] = [
                pd.Timestamp(today - timedelta(days=(N-1-i))) for i in range(N)
            ]

        if run:
            parts = []
            alert_needed = False
            alert_names = []

            for name, g in df.groupby("Nama", sort=False):
                g2 = g.sort_values("Tanggal").reset_index(drop=True)
                g2 = detect_anomaly_df(g2)

                pred_s = rk4_predict_series(g2["Systolic"])
                pred_d = rk4_predict_series(g2["Diastolic"])

                g2["Prediksi_Systolic"] = np.nan
                g2["Prediksi_Diastolic"] = np.nan
                if pred_s is not None:
                    g2.at[len(g2)-1,"Prediksi_Systolic"] = pred_s
                    g2.at[len(g2)-1,"Prediksi_Diastolic"] = pred_d

                parts.append(g2)

                if g2["Anom_Total"].any():
                    alert_needed = True
                    alert_names.append(name)

            result = pd.concat(parts, ignore_index=True)

            st.subheader("Hasil Analisis RK4")
            st.dataframe(result)

            st.session_state.last_result = result
            st.session_state.last_context = {"mode":"Input","file":uploaded.name}

            # ========== POPUP DRAMATIC FIXED ==========
            if alert_needed:
                st.error(f"🚨 Anomali terdeteksi pada: {', '.join(alert_names[:6])}")
                warning_popup()

                # Siren sound
                wav = generate_siren_wav()
                datauri = wav_bytes_to_datauri(wav)
                st.markdown(
                    f"""<audio autoplay><source src="{datauri}" type="audio/wav"></audio>""",
                    unsafe_allow_html=True
                )

            else:
                st.success("✔ Tidak ada hipertensi/hipotensi terdeteksi.")
                normal_popup()

                # Ting sound
                wav = generate_ting_wav()
                datauri = wav_bytes_to_datauri(wav)
                st.markdown(
                    f"""<audio autoplay><source src="{datauri}" type="audio/wav"></audio>""",
                    unsafe_allow_html=True
                )

    if st.button("⬅ Kembali"):
        st.session_state.page = "beranda"

    st.stop()
# ============================================================
# ==================   PERSONAL ANALYSIS   ====================
# ============================================================

if st.session_state.page == "personal":

    st.header("👤 Analisis Personal")

    name = st.text_input("Nama")
    n = st.number_input("Jumlah data tensi (1–10)", 1, 10, 4)

    systolic = []
    diastolic = []

    for i in range(int(n)):
        c1, c2 = st.columns(2)
        with c1:
            s = st.number_input(f"Sistolik #{i+1}", 40, 250, 120, key=f"s_{i}")
        with c2:
            d = st.number_input(f"Diastolik #{i+1}", 20, 180, 80, key=f"d_{i}")
        systolic.append(s)
        diastolic.append(d)

    analyze = st.button("Analisis (RK4)")

    if analyze:

        if not name:
            st.error("Nama wajib diisi.")
            st.stop()

        dfp = pd.DataFrame({
            "Nama": [name]*int(n),
            "Tanggal": [
                pd.Timestamp(datetime.now().date()-timedelta(days=(int(n)-1-i)))
                for i in range(int(n))
            ],
            "Systolic": systolic,
            "Diastolic": diastolic
        })

        dfp = detect_anomaly_df(dfp)

        pred_s = rk4_predict_series(dfp["Systolic"])
        pred_d = rk4_predict_series(dfp["Diastolic"])

        dfp["Prediksi_Systolic"] = np.nan
        dfp["Prediksi_Diastolic"] = np.nan
        if pred_s is not None:
            dfp.at[len(dfp)-1,"Prediksi_Systolic"] = pred_s
            dfp.at[len(dfp)-1,"Prediksi_Diastolic"] = pred_d

        st.subheader("Hasil Analisis Personal")
        st.dataframe(dfp)

        # Grafik
        fig, ax = plt.subplots(figsize=(9,3))
        ax.plot(dfp["Tanggal"], dfp["Systolic"], marker="o", label="Systolic")
        ax.plot(dfp["Tanggal"], dfp["Diastolic"], marker="o", label="Diastolic")
        if pred_s is not None:
            nd = dfp["Tanggal"].iloc[-1] + timedelta(days=1)
            ax.scatter([nd],[pred_s], marker="D", s=80)
            ax.scatter([nd],[pred_d], marker="D", s=80)
        ax.set_title(f"Tensi — {name}")
        ax.legend()
        st.pyplot(fig)

        # ========== POPUP DRAMATIC FIXED ==========
        if dfp["Anom_Total"].iloc[-1]:
            st.error("⚠️ Terdeteksi hipertensi / hipotensi pada data terakhir!")
            warning_popup()

            # Siren
            wav = generate_siren_wav()
            datauri = wav_bytes_to_datauri(wav)
            st.markdown(
                f"""<audio autoplay><source src="{datauri}" type="audio/wav"></audio>""",
                unsafe_allow_html=True
            )
        else:
            st.success("✔ Datamu normal. Jaga kesehatan yaaa!")
            normal_popup()

            # Ting
            wav = generate_ting_wav()
            datauri = wav_bytes_to_datauri(wav)
            st.markdown(
                f"""<audio autoplay><source src="{datauri}" type="audio/wav"></audio>""",
                unsafe_allow_html=True
            )

        if pred_s is not None:
            st.markdown(
                f"**Prediksi RK4 (1 langkah berikutnya):** "
                f"Systolic **{pred_s:.2f}**, Diastolic **{pred_d:.2f}**"
            )

        st.session_state.last_result = dfp
        st.session_state.last_context = {"mode":"Personal","name":name}

    if st.button("⬅ Kembali"):
        st.session_state.page = "beranda"

    st.stop()


# ============================================================
# ==================   HALAMAN HASIL ANALISIS   ==============
# ============================================================

if st.session_state.page == "hasil":

    st.header("📊 Hasil Analisis Terakhir")

    if st.session_state.last_result is None:
        st.info("Belum ada hasil analisis.")
    else:
        st.write("Konteks Analisis:", st.session_state.last_context)
        st.dataframe(st.session_state.last_result)

        df_show = st.session_state.last_result
        total_anom = int(df_show["Anom_Total"].sum()) if "Anom_Total" in df_show.columns else 0

        st.markdown(f"**Total hipertensi/hipotensi terdeteksi: {total_anom}**")

    if st.button("⬅ Kembali"):
        st.session_state.page = "beranda"

    st.stop()


# ============================================================
# =====================   MENGAPA RK4?   ======================
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
# =======================   FOOTER   ==========================
# ============================================================

st.markdown("---")
st.markdown(
    "<div style='color:var(--muted); font-size:13px;'>"
    "NADI | RK4 — Aplikasi edukasi komputasi tekanan darah (bukan alat diagnosis medis)."
    "</div>",
    unsafe_allow_html=True
)
