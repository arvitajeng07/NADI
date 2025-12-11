# app.py
"""
Sistem Deteksi Anomali & Prediksi Tensi (RK4)
Tema: Biru - Putih
Fitur lengkap:
 - Landing page (dua "kantong" pilih mode: Input Data / Personal)
 - ANALISIS DATA: upload CSV/XLSX (kolom minimal: Nama, Systolic, Diastolic, Tanggal optional) -> RK4 -> alarm overlay
 - PERSONAL: input manual sampai 10 data -> RK4 -> alarm overlay
 - HALAMAN: Hasil Analisis (tampilkan hasil terakhir)
 - HALAMAN: Mengapa RK4? (penjelasan singkat)
 - Alarm otomatis + full-screen siren overlay + tombol hentikan
Requirements:
 - streamlit, pandas, numpy, matplotlib, soundfile, openpyxl, pillow
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import soundfile as sf
from datetime import datetime, timedelta
import streamlit.components.v1 as components

# -----------------------
# Page config and initial session state
# -----------------------
st.set_page_config(page_title="TensiCheck (RK4)", layout="wide", initial_sidebar_state="expanded")

if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_context" not in st.session_state:
    st.session_state.last_context = None

# -----------------------
# Styling (blue-white)
# -----------------------
st.markdown(
    """
    <style>
    :root {
      --primary-1: #0b63d9;
      --primary-2: #1ea7ff;
      --muted: #6b7280;
      --card-bg: linear-gradient(180deg, #ffffff, #f6fbff);
    }
    .title { font-size:32px; font-weight:800; color:var(--primary-1); }
    .subtitle { color:var(--muted); margin-bottom:8px; }
    .card { background: var(--card-bg); padding:18px; border-radius:12px; box-shadow: 0 6px 20px rgba(11,99,217,0.06); }
    .mode-tile { border-radius:12px; padding:18px; text-align:center; transition: transform .12s ease-in-out; cursor:pointer; }
    .mode-tile:hover { transform: translateY(-6px); box-shadow: 0 12px 30px rgba(11,99,217,0.08); }
    .cta { background: linear-gradient(90deg,var(--primary-1), var(--primary-2)); color:white; padding:10px 18px; border-radius:10px; font-weight:700; border:none; }
    .muted { color:var(--muted); }
    /* siren overlay */
    .siren-overlay {
      position: fixed;
      inset: 0;
      z-index: 9999;
      display: flex;
      align-items: center;
      justify-content: center;
      background: rgba(0,0,0,0.66);
      backdrop-filter: blur(3px);
    }
    .siren-box {
      text-align:center;
      color: white;
      padding: 24px;
      border-radius: 12px;
    }
    .siren-svg { width: 48vmin; height: 48vmin; animation: pulse 1s infinite; }
    @keyframes pulse {
      0% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.03); opacity: 0.95; } 100% { transform: scale(1); opacity: 1; }
    }
    .stop-btn {
      margin-top: 18px;
      background: white;
      color: #c53030;
      padding: 10px 20px;
      border-radius: 8px;
      font-weight:700;
      border:none;
      cursor:pointer;
    }
    </style>
    """, unsafe_allow_html=True
)

# -----------------------
# Helpers: audio (siren), datauri
# -----------------------
def generate_siren_wav(duration=1.8, sr=44100):
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    mod = 0.5 * (1 + np.sin(2 * np.pi * 1.2 * t))
    freq = 800 + 300 * np.sin(2 * np.pi * 0.6 * t)
    tone = 0.6 * np.sin(2 * np.pi * freq * t) * (0.6 + 0.4 * mod)
    tone = np.clip(tone, -1, 1)
    buf = BytesIO()
    sf.write(buf, tone, sr, format='WAV')
    buf.seek(0)
    return buf.read()

def wav_bytes_to_datauri(wav_bytes):
    b64 = base64.b64encode(wav_bytes).decode()
    return f"data:audio/wav;base64,{b64}"

# -----------------------
# RK4 helpers
# -----------------------
def rk4_predict_value(last, prev, h=1.0):
    slope = last - prev
    def f(t, y):
        return slope
    k1 = f(0, last)
    k2 = f(h/2, last + h*k1/2)
    k3 = f(h/2, last + h*k2/2)
    k4 = f(h, last + h*k3)
    return last + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def rk4_predict_series(arr):
    arr = list(arr)
    if len(arr) < 2:
        return None
    return rk4_predict_value(arr[-1], arr[-2])

# -----------------------
# Anomaly detection
# -----------------------
def detect_anomaly_df(df, thresh_sys_high=140, thresh_dia_high=90, thresh_sys_low=90, thresh_dia_low=60):
    df = df.copy()
    df['Systolic'] = pd.to_numeric(df['Systolic'], errors='coerce')
    df['Diastolic'] = pd.to_numeric(df['Diastolic'], errors='coerce')
    # Thresholds (default lebih sensitif: 140/90 untuk hipertensi)
    df['Anom_Threshold'] = (
        (df['Systolic'] >= thresh_sys_high) | (df['Diastolic'] >= thresh_dia_high) |
        (df['Systolic'] <= thresh_sys_low) | (df['Diastolic'] <= thresh_dia_low)
    )
    # Z-score
    for col in ['Systolic','Diastolic']:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            df[f'Z_{col[:3]}'] = 0.0
        else:
            df[f'Z_{col[:3]}'] = (df[col] - mean) / std
    df['Anom_Z'] = (df['Z_Sys'].abs() > 2.5) | (df['Z_Dia'].abs() > 2.5)
    # sudden jump
    df['Delta_Sys'] = df['Systolic'].diff().abs().fillna(0)
    df['Delta_Dia'] = df['Diastolic'].diff().abs().fillna(0)
    df['Anom_Jump'] = (df['Delta_Sys'] > 20) | (df['Delta_Dia'] > 15)
    df['Anom_Total'] = df['Anom_Threshold'] | df['Anom_Z'] | df['Anom_Jump']
    return df

# -----------------------
# Siren overlay render (HTML with audio autoplay & stop button)
# -----------------------
def render_siren_overlay(audio_datauri, title_text="PERINGATAN: Anomali Tensi Terdeteksi"):
    html = f"""
    <div class="siren-overlay" id="sirenOverlay">
      <div class="siren-box" role="dialog" aria-live="assertive">
        <svg class="siren-svg" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
          <circle cx="60" cy="60" r="56" fill="#ff4d4d"/>
          <g transform="translate(20,22)">
            <rect width="80" height="60" rx="10" fill="#fff"/>
            <rect x="10" y="6" width="60" height="6" rx="3" fill="#ff4d4d"/>
            <circle cx="40" cy="36" r="10" fill="#ff4d4d"/>
            <circle cx="40" cy="36" r="6" fill="#fff"/>
          </g>
        </svg>
        <h1 style="font-size:32px; margin:8px 0; color:white;">{title_text}</h1>
        <p style="color:rgba(255,255,255,0.9); margin:0 0 12px;">Sistem mendeteksi kemungkinan hipertensi atau hipotensi. Periksa data & tindakan klinis diperlukan.</p>
        <button class="stop-btn" onclick="document.getElementById('alarmAudio').pause(); document.getElementById('sirenOverlay').style.display='none'">Hentikan Alarm</button>
        <audio id="alarmAudio" autoplay loop>
          <source src="{audio_datauri}" type="audio/wav">
        </audio>
      </div>
    </div>
    """
    components.html(html, height=150)

# -----------------------
# Sidebar & Navigation
# -----------------------
st.sidebar.title("Menu")
page = st.sidebar.radio("Pilih halaman", ("Beranda", "Input Data (Puskesmas)", "Personal", "Hasil Analisis", "Mengapa RK4?"))

# -----------------------
# Landing: Judul + Sambutan + Dua Kantong (tiles)
# -----------------------
if page == "Beranda":
    st.markdown('<div class="nadi-title">NADI : Numeric Analysis of Diastolic & Systolic</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="nadi-desc">
        <b>Adalah ruang sederhana untuk membaca alur tekanan darah Anda melalui pendekatan komputasi.</b>
        Dengan memanfaatkan metode <b>RK4</b> dan proses pengkodingan yang turut terbantu oleh kecerdasan buatan, <b>NADI</b> menghadirkan analisis yang ringan, intuitif, dan mudah dipahami.
        <br><br>
        Namun, <b>NADI bukan alat diagnosis medis</b>. Hasil yang ditampilkan hanya gambaran komputasi, bukan pengganti konsultasi tenaga kesehatan profesional.
        Gunakan <b>NADI</b> sebagai langkah awal untuk mengenali pola, bukan sebagai keputusan akhir kesehatan Anda.
        <br><br>
        <i>Selamat datang. Biarkan NADI membaca aliran kesehatan Anda.</i>
        </div>
        """,
        unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 style="color:#0b63d9; margin-bottom:6px;">Input Data (Puskesmas / Klinik)</h3>', unsafe_allow_html=True)
        st.write("Untuk analisis populasi: upload CSV/XLSX berisi banyak pasien. Sistem akan memproses tiap pasien, menandai anomali, dan memprediksi tekanan berikutnya.")
        st.markdown('<div style="margin-top:8px;"><a class="cta" href="#" onclick="window.location.hash=\'/Input Data (Puskesmas)\'; location.reload()">Masuk ke Input Data</a></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3 style="color:#0b63d9; margin-bottom:6px;">Analisis Personal</h3>', unsafe_allow_html=True)
        st.write("Untuk analisis personal: input sampai 10 data tensi untuk prediksi dan deteksi anomali.")
        st.markdown('<div style="margin-top:8px;"><a class="cta" href="#" onclick="window.location.hash=\'/Personal\'; location.reload()">Masuk ke Personal</a></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Contoh Template Data (untuk upload)")
    st.write("File harus minimal memiliki kolom: Nama, Systolic, Diastolic. Tanggal bersifat opsional.")
    sample_df = pd.DataFrame({
        "Nama": ["Budi","Budi","Siti","Siti"],
        "Tanggal": [datetime.now().date()-timedelta(days=3), datetime.now().date()-timedelta(days=1),
                    datetime.now().date()-timedelta(days=2), datetime.now().date()],
        "Systolic": [120, 145, 130, 170],
        "Diastolic": [80, 95, 85, 105]
    })
    st.dataframe(sample_df)
    csv = sample_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download contoh CSV", data=csv, file_name="template_tensi.csv", mime="text/csv")

# -----------------------
# Input Data (multi-pasien)
# -----------------------
elif page == "Input Data (Puskesmas)":
    st.markdown('<div class="title">Analisis Data Populasi</div>', unsafe_allow_html=True)
    st.write("Upload file CSV atau XLSX. Pastikan kolom: Nama, Systolic, Diastolic. Tanggal optional (dipakai untuk grafik jika ada).")

    uploaded = st.file_uploader("Upload CSV / XLSX", type=["csv","xlsx"])
    run = st.button("Analisis (RK4)")

    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            st.stop()

        df.columns = [c.strip() for c in df.columns]
        st.info("Pratinjau data (baris awal). Pastikan kolom ada.")
        st.dataframe(df.head(50))

        required = {"Nama","Systolic","Diastolic"}
        if not required.issubset(set(df.columns)):
            st.error(f"File harus berisi kolom minimal: {sorted(list(required))}")
        else:
            if "Tanggal" in df.columns:
                df['Tanggal'] = pd.to_datetime(df['Tanggal'], errors='coerce')
                if df['Tanggal'].isna().any():
                    df['Tanggal'] = df['Tanggal'].fillna(method='ffill').fillna(pd.Timestamp(datetime.now()))
            else:
                n = len(df)
                today = datetime.now().date()
                df['Tanggal'] = [pd.Timestamp(today - timedelta(days=(n-1-i))) for i in range(n)]

            if run:
                parts = []
                siren_needed = False
                siren_names = []
                for name, g in df.groupby('Nama', sort=False):
                    g2 = g.sort_values('Tanggal').reset_index(drop=True)
                    g2 = g2[['Nama','Tanggal','Systolic','Diastolic']].copy()
                    g2 = detect_anomaly_df(g2)  # default thresholds
                    pred_sys = rk4_predict_series(g2['Systolic'].values)
                    pred_dia = rk4_predict_series(g2['Diastolic'].values)
                    g2['Prediksi_Systolic'] = np.nan
                    g2['Prediksi_Diastolic'] = np.nan
                    if pred_sys is not None:
                        g2.at[g2.index[-1], 'Prediksi_Systolic'] = pred_sys
                        g2.at[g2.index[-1], 'Prediksi_Diastolic'] = pred_dia
                    parts.append(g2)
                    if g2['Anom_Total'].any() if 'Anom_Total' in g2.columns else g2['Anom_Total'].any():
                        pass  # placeholder (we use consistent 'Anom_Total' below)
                    if g2['Anom_Total'].any():
                        siren_needed = True
                        siren_names.append(name)
                result = pd.concat(parts, ignore_index=True)
                st.subheader("Hasil Analisis (ringkasan)")
                st.dataframe(result)

                # save to session for Hasil Analisis page
                st.session_state.last_result = result
                st.session_state.last_context = {"mode":"Input Data", "file": uploaded.name}

                total_anom = int(result['Anom_Total'].sum())
                if total_anom > 0:
                    st.error(f"🚨 TERDETEKSI {total_anom} data anomali. Contoh pasien: {', '.join(map(str,siren_names[:6]))}")
                    wav = generate_siren_wav(duration=2.0)
                    datauri = wav_bytes_to_datauri(wav)
                    render_siren_overlay(datauri, title_text="PERINGATAN: Anomali Tensi Terdeteksi!")
                    st.subheader("Detail pasien bermasalah")
                    for p in siren_names:
                        with st.expander(f"Pasien: {p}"):
                            sub = result[result['Nama']==p].reset_index(drop=True)
                            st.dataframe(sub)
                            fig, ax = plt.subplots(figsize=(8,3))
                            ax.plot(sub['Tanggal'], sub['Systolic'], marker='o', label='Systolic')
                            ax.plot(sub['Tanggal'], sub['Diastolic'], marker='o', label='Diastolic')
                            ax.scatter(sub[sub['Anom_Total']]['Tanggal'], sub[sub['Anom_Total']]['Systolic'], color='red', marker='x', s=90, label='Anomali')
                            ax.set_title(f"Tensi - {p}")
                            ax.legend()
                            st.pyplot(fig)
                else:
                    st.success("✔ Tidak ada anomali terdeteksi pada file ini.")
                    patients = result['Nama'].unique().tolist()
                    sel = st.selectbox("Pilih pasien untuk lihat detail", patients)
                    sub = result[result['Nama']==sel].reset_index(drop=True)
                    st.dataframe(sub)
                    fig, ax = plt.subplots(figsize=(8,3))
                    ax.plot(sub['Tanggal'], sub['Systolic'], marker='o', label='Systolic')
                    ax.plot(sub['Tanggal'], sub['Diastolic'], marker='o', label='Diastolic')
                    ax.set_title(f"Tensi - {sel}")
                    ax.legend()
                    st.pyplot(fig)

# -----------------------
# Page: Personal
# -----------------------
elif page == "Personal":
    st.markdown('<div class="title">Analisis Personal</div>', unsafe_allow_html=True)
    st.write("Masukkan nama, pilih jumlah data (max 10), isi pasangan Systolic & Diastolic, lalu klik Analisis (RK4).")
    colL, colR = st.columns([2,1])
    with colL:
        name = st.text_input("Nama")
        n = st.number_input("Jumlah data (1-10)", min_value=1, max_value=10, value=4, step=1)
        systolic = []
        diastolic = []
        for i in range(int(n)):
            c1, c2 = st.columns(2)
            with c1:
                s = st.number_input(f"Sistolik #{i+1}", min_value=40, max_value=250, value=120, key=f"ps_{i}")
            with c2:
                d = st.number_input(f"Diastolik #{i+1}", min_value=20, max_value=180, value=80, key=f"pd_{i}")
            systolic.append(s)
            diastolic.append(d)
        analyze = st.button("Analisis (RK4)")
    with colR:
        st.markdown('<div class="card"><h4 style="color:#0b63d9;">Petunjuk</h4><p class="muted">Masukkan data dalam urutan kronologis (yang paling awal dahulu). Prediksi akan dilakukan 1 langkah ke depan berdasarkan dua data terakhir menggunakan RK4 sederhana.</p></div>', unsafe_allow_html=True)

    if analyze:
        if not name:
            st.error("Masukkan nama terlebih dahulu.")
        else:
            dfp = pd.DataFrame({
                "Nama":[name]*int(n),
                "Tanggal":[pd.Timestamp(datetime.now().date()-timedelta(days=(int(n)-1-i))) for i in range(int(n))],
                "Systolic":systolic,
                "Diastolic":diastolic
            })
            dfp = detect_anomaly_df(dfp)
            pred_s = rk4_predict_series(dfp['Systolic'].values)
            pred_d = rk4_predict_series(dfp['Diastolic'].values)
            if pred_s is not None:
                dfp.at[dfp.index[-1],'Prediksi_Systolic'] = pred_s
                dfp.at[dfp.index[-1],'Prediksi_Diastolic'] = pred_d
            st.subheader("Hasil Analisis Personal")
            st.dataframe(dfp)

            # save to session
            st.session_state.last_result = dfp
            st.session_state.last_context = {"mode":"Personal", "name": name}

            # chart
            fig, ax = plt.subplots(figsize=(9,3))
            ax.plot(dfp['Tanggal'], dfp['Systolic'], marker='o', label='Systolic')
            ax.plot(dfp['Tanggal'], dfp['Diastolic'], marker='o', label='Diastolic')
            if pred_s is not None:
                nd = dfp['Tanggal'].iloc[-1] + pd.Timedelta(days=1)
                ax.scatter([nd],[pred_s], marker='D', s=80, label='Prediksi Systolic')
                ax.scatter([nd],[pred_d], marker='D', s=80, label='Prediksi Diastolic')
            ax.set_title(f"Tensi - {name}")
            ax.legend()
            st.pyplot(fig)

            # check last data anomaly
            last = dfp.iloc[-1]
            if bool(last['Anom_Total']):
                st.error("⚠️ Terdeteksi anomali pada data terakhir!")
                wav = generate_siren_wav(duration=2.0)
                datauri = wav_bytes_to_datauri(wav)
                render_siren_overlay(datauri, title_text="PERINGATAN: Anomali pada Data!")
            else:
                st.success("✔ Data terakhir tampak normal.")

            # show predictions
            if pred_s is not None:
                st.markdown(f"**Prediksi RK4 (1 langkah ke depan)** — Sistolik: **{pred_s:.2f}**, Diastolik: **{pred_d:.2f}**")

# -----------------------
# Page: Hasil Analisis (lihat hasil terakhir)
# -----------------------
elif page == "Hasil Analisis":
    st.markdown('<div class="title">Hasil Analisis Terakhir</div>', unsafe_allow_html=True)
    if st.session_state.last_result is None:
        st.info("Belum ada hasil analisis. Lakukan analisis pada halaman Input Data atau Personal terlebih dahulu.")
    else:
        st.write("Context:", st.session_state.last_context)
        st.dataframe(st.session_state.last_result)
        # quick summary
        df_show = st.session_state.last_result
        total_anom = int(df_show['Anom_Total'].sum()) if 'Anom_Total' in df_show.columns else 0
        st.markdown(f"**Total anomali terdeteksi:** {total_anom}")

# -----------------------
# Page: Mengapa RK4?
# -----------------------
elif page == "Mengapa RK4?":
    st.markdown('<div class="title">Mengapa Runge–Kutta Orde-4 (RK4)?</div>', unsafe_allow_html=True)
    st.write("Ringkasan singkat kenapa memakai RK4 untuk prediksi sederhana:")
    st.markdown("""
    - **RK4** adalah metode numerik berorde tinggi yang sering dipakai untuk menyelesaikan persamaan diferensial biasa (ODE).
    - Untuk data time-series seperti tekanan darah, RK4 memberikan estimasi yang lebih stabil dibanding metode sederhana (mis. Euler) karena mengkombinasikan beberapa evaluasi turunan (k1..k4).
    - Dalam aplikasi demo ini, kita mengestimasi turunan (perubahan) dari dua titik terakhir dan menjalankan satu langkah RK4 — hasilnya seringkali *lebih halus* dan kurang berisik daripada pendekatan 1-step sederhana.
    - **Catatan klinis:** prediksi ini bersifat ilustrasi. Untuk keputusan medis, gunakan model yang divalidasi klinis dan data kontekstual (obat, riwayat, aktivitas, dsb.).
    """)
    st.markdown("**Bagan singkat (intuisi):**")
    st.image("https://upload.wikimedia.org/wikipedia/commons/1/1a/Runge-Kutta_method.svg", caption="Sketsa Runge-Kutta (ilustrasi)")

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.markdown('<div class="muted">Catatan: Aplikasi ini untuk tujuan demo/pendukung edukasi — bukan pengganti konsultasi medis. Sesuaikan aturan deteksi dan threshold jika diperlukan.</div>', unsafe_allow_html=True)
