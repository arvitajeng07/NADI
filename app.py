# app.py
"""
APP.PY FINAL — NADI (RK4) with dramatic popup + siren overlay
Style: Soft Aesthetic Blue
Features:
 - Landing page (tombol navigasi)
 - Upload CSV/XLSX analysis (multi-patient) -> RK4 -> anomaly detection -> dramatic popup -> siren overlay
 - Personal input (1-10 data) -> RK4 -> anomaly detection -> dramatic popup -> siren overlay
 - Hasil Analisis (last result)
 - Mengapa RK4?
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
import time

# -----------------------
# Page config & initial state
# -----------------------
st.set_page_config(page_title="NADI (RK4) — Soft Blue", layout="wide")
if "page" not in st.session_state:
    st.session_state.page = "beranda"
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_context" not in st.session_state:
    st.session_state.last_context = None

# -----------------------
# Soft Aesthetic CSS + small modal + dramatic popup CSS/JS
# -----------------------
st.markdown(
    """
    <style>
    /* Font + theme */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    :root{
      --bg1: #f3f9ff; --bg2: #eaf6ff;
      --primary-1: #0b63d9; --primary-2: #1ea7ff; --muted: #6b7280;
    }
    html, body, [class*="css"]  {
      font-family: Inter, "Times New Roman", serif;
      background: linear-gradient(180deg, var(--bg1), var(--bg2));
    }

    /* Header */
    .big-nadi-title { font-family: "Times New Roman", serif; font-size: 52px; font-weight:700; color:var(--primary-1); text-align:center; margin-top:6px; margin-bottom:6px; }
    .nadi-desc { text-align:center; color:var(--muted); max-width:920px; margin:auto; margin-bottom:18px; font-size:15.5px; line-height:1.45; }

    /* Glass card + tiles */
    .glass { background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(246,251,255,0.88)); border-radius:14px; padding:18px; box-shadow:0 8px 28px rgba(11,99,217,0.06); border:1px solid rgba(255,255,255,0.6); transition: transform .18s ease, box-shadow .18s ease; }
    .glass:hover { transform: translateY(-6px); box-shadow:0 18px 40px rgba(11,99,217,0.09); }
    .cta-btn { display:inline-block; padding:10px 20px; border-radius:12px; background: linear-gradient(90deg, var(--primary-1), var(--primary-2)); color:white; font-weight:700; border:none; box-shadow:0 8px 18px rgba(11,99,217,0.12); cursor:pointer; }
    .spacer { margin-top:14px; margin-bottom:14px; }

    /* small modal warning */
    #modal-warning { position: fixed; top:0; left:0; width:100%; height:100%; display:none; align-items:center; justify-content:center; background: rgba(4,8,13,0.45); z-index:99999; }
    .modal-content { width:360px; background:white; border-radius:12px; padding:22px; text-align:center; box-shadow:0 10px 30px rgba(2,6,23,0.12); animation: pop .22s ease-out; }
    @keyframes pop { from {transform:scale(.92);opacity:0} to {transform:scale(1);opacity:1} }
    .alarm-img-small { width:72px; margin-bottom:10px; }

    /* dramatic big popup */
    #dramatic-warning {
      position: fixed; top:0; left:0; width:100%; height:100%;
      background: rgba(0,0,0,0.78); backdrop-filter: blur(6px);
      display:none; align-items:center; justify-content:center; z-index:99998;
    }
    .dramatic-box {
      background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
      padding: 48px 64px; border-radius:22px; text-align:center;
      box-shadow: 0 0 80px rgba(255,0,0,0.18);
      animation: zoomIn .3s ease-out, shake .35s .05s ease-in-out;
      border: 1px solid rgba(255,255,255,0.08);
      max-width: 780px;
    }
    .dramatic-icon { width: 140px; filter: drop-shadow(0 0 12px rgba(255,0,0,0.6)); margin-bottom:18px; }
    #dramatic-text { font-size:46px; font-weight:900; color:#ff1e1e; text-shadow:0 0 20px rgba(255,0,0,0.7); margin-bottom:8px; }
    #dramatic-countdown { font-size:32px; font-weight:700; color:white; margin-top:10px; }

    @keyframes zoomIn { 0%{transform:scale(.6);opacity:0} 100%{transform:scale(1);opacity:1} }
    @keyframes shake { 0%{transform:translateX(-6px)} 35%{transform:translateX(6px)} 70%{transform:translateX(-3px)} 100%{transform:translateX(0)} }

    /* siren overlay styles */
    .siren-overlay{ position: fixed; inset:0; z-index:999999; display:flex; align-items:center; justify-content:center; background: rgba(0,0,0,0.66); backdrop-filter: blur(3px); }
    .siren-box { text-align:center; color:white; padding:24px; border-radius:14px; }
    .siren-svg { width: 34vmin; height: 34vmin; animation: pulse 1s infinite; }
    @keyframes pulse { 0%{transform:scale(1);opacity:1}50%{transform:scale(1.03);opacity:0.95}100%{transform:scale(1);opacity:1} }
    .stop-btn { margin-top:18px; background:white; color:#c53030; padding:10px 18px; border-radius:10px; font-weight:700; border:none; cursor:pointer; }

    </style>

    <!-- small modal HTML -->
    <div id="modal-warning">
      <div class="modal-content">
        <img class="alarm-img-small" src="https://cdn-icons-png.flaticon.com/512/463/463612.png">
        <div id="modal-text" style="font-weight:700; color:#0b63d9;">Peringatan</div>
      </div>
    </div>

    <!-- dramatic popup HTML -->
    <div id="dramatic-warning">
      <div class="dramatic-box">
        <img class="dramatic-icon" src="https://cdn-icons-png.flaticon.com/512/463/463612.png">
        <div id="dramatic-text">PERINGATAN ANOMALI</div>
        <div id="dramatic-countdown">Alarm aktif dalam 3...</div>
      </div>
    </div>

    <script>
    function showWarning(text){
        try{
          const el = window.parent.document.querySelector('#modal-warning') || document.getElementById('modal-warning');
          if(el){
            const txt = el.querySelector('#modal-text');
            if(txt) txt.innerText = text;
            el.style.display = 'flex';
            setTimeout(()=>{ el.style.display = 'none'; }, 1700);
          } else {
            alert(text);
          }
        }catch(e){
          alert(text);
        }
    }

    // dramatic popup controller
    function showDramaticWarning(text){
        try{
            const box = window.parent.document.querySelector('#dramatic-warning') || document.getElementById('dramatic-warning');
            const t = box.querySelector('#dramatic-text');
            const c = box.querySelector('#dramatic-countdown');
            t.innerText = text;
            box.style.display = 'flex';
            let countdown = 3;
            c.innerText = "Alarm aktif dalam " + countdown + "...";
            const id = setInterval(()=>{
                countdown -= 1;
                if(countdown > 0){
                    c.innerText = "Alarm aktif dalam " + countdown + "...";
                } else {
                    clearInterval(id);
                }
            }, 700);
            // hide automatically after ~2.1s
            setTimeout(()=>{ box.style.display = 'none'; }, 2100);
        }catch(e){
            console.log("showDramaticWarning err", e);
            alert(text);
        }
    }
    </script>
    """,
    unsafe_allow_html=True
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
# RK4 helpers (original)
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
# Anomaly detection (original)
# -----------------------
def detect_anomaly_df(df, thresh_sys_high=140, thresh_dia_high=90, thresh_sys_low=90, thresh_dia_low=60):
    df = df.copy()
    df['Systolic'] = pd.to_numeric(df['Systolic'], errors='coerce')
    df['Diastolic'] = pd.to_numeric(df['Diastolic'], errors='coerce')
    df['Anom_Threshold'] = (
        (df['Systolic'] >= thresh_sys_high) | (df['Diastolic'] >= thresh_dia_high) |
        (df['Systolic'] <= thresh_sys_low) | (df['Diastolic'] <= thresh_dia_low)
    )
    for col in ['Systolic','Diastolic']:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            df[f'Z_{col[:3]}'] = 0.0
        else:
            df[f'Z_{col[:3]}'] = (df[col] - mean) / std
    df['Anom_Z'] = (df['Z_Sys'].abs() > 2.5) | (df['Z_Dia'].abs() > 2.5)
    df['Delta_Sys'] = df['Systolic'].diff().abs().fillna(0)
    df['Delta_Dia'] = df['Diastolic'].diff().abs().fillna(0)
    df['Anom_Jump'] = (df['Delta_Sys'] > 20) | (df['Delta_Dia'] > 15)
    df['Anom_Total'] = df['Anom_Threshold'] | df['Anom_Z'] | df['Anom_Jump']
    return df

# -----------------------
# Siren overlay render (full-screen)
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
# NAV: BERANDA (tombol)
# -----------------------
if st.session_state.page == "beranda":
    st.markdown("<div class='big-nadi-title'>✨ NADI : Numeric Analysis of Diastolic & Systolic</div>", unsafe_allow_html=True)
    st.markdown("<div class='nadi-desc'><b>Adalah ruang sederhana untuk membaca alur tekanan darah Anda melalui pendekatan komputasi.</b><br>Dengan memanfaatkan metode <b>RK4</b> dan proses pengkodingan yang turut terbantu oleh kecerdasan buatan, <b>NADI</b> menghadirkan analisis yang ringan, intuitif, dan mudah dipahami.<br><br>Namun, <b>NADI bukan alat diagnosis medis</b>. Hasil yang ditampilkan hanya gambaran komputasi, bukan pengganti konsultasi tenaga kesehatan profesional.<br><br><i>Selamat datang. Biarkan NADI membaca aliran kesehatan Anda.</i></div>", unsafe_allow_html=True)

    # two main tiles
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="glass"><h3 style="color:var(--primary-1)">Input Data (Puskesmas / Klinik)</h3><p style="color:var(--muted)">Untuk analisis populasi: upload CSV/XLSX berisi banyak pasien. Sistem akan memproses tiap pasien, menandai anomali, dan memprediksi tekanan berikutnya.</p><div class="spacer"></div><div><button class="cta-btn" onclick="(function(){window.dispatchEvent(new CustomEvent(\'setPage\', {detail:\'input\'}));})()">Masuk ke Input Data</button></div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="glass"><h3 style="color:var(--primary-1)">Analisis Personal</h3><p style="color:var(--muted)">Untuk analisis personal: input sampai 10 data tensi untuk prediksi dan deteksi anomali.</p><div class="spacer"></div><div><button class="cta-btn" onclick="(function(){window.dispatchEvent(new CustomEvent(\'setPage\', {detail:\'personal\'}));})()">Masuk ke Personal</button></div></div>', unsafe_allow_html=True)

    st.write("")
    # extra horizontal buttons
    col_a, col_b, col_c = st.columns([1,1,1])
    with col_a:
        if st.button("📊 Lihat Hasil Analisis Terakhir"):
            st.session_state.page = "hasil"
    with col_b:
        if st.button("❔ Mengapa RK4?"):
            st.session_state.page = "rk4info"
    with col_c:
        if st.button("🔄 Reset hasil terakhir"):
            st.session_state.last_result = None
            st.session_state.last_context = None
            st.success("Riwayat hasil direset.")

    st.markdown("---")
    st.subheader("Contoh Template Data (untuk upload)")
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

    # JS listener to set page via custom event (for HTML buttons)
    st.markdown("""
    <script>
    window.addEventListener('setPage', function(e){
        const page = e.detail;
        // try to set Streamlit page by changing location.hash (fallback)
        window.location.hash = page;
    });
    </script>
    """, unsafe_allow_html=True)

    st.stop()

# -----------------------
# HALAMAN: INPUT DATA (Puskesmas)
# -----------------------
if st.session_state.page == "input":
    st.header("📁 Analisis Data Populasi (Upload CSV / XLSX)")
    uploaded = st.file_uploader("Upload CSV / XLSX (kolom minimal: Nama, Systolic, Diastolic, Tanggal opsional)", type=["csv","xlsx"])
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
                    if g2.get('Anom_Total', pd.Series([False]*len(g2))).any():
                        siren_needed = True
                        siren_names.append(name)
                result = pd.concat(parts, ignore_index=True)
                st.subheader("Hasil Analisis (ringkasan)")
                st.dataframe(result)

                # save to session for Hasil Analisis page
                st.session_state.last_result = result
                st.session_state.last_context = {"mode":"Input Data", "file": uploaded.name}

                total_anom = int(result['Anom_Total'].sum()) if 'Anom_Total' in result.columns else 0
                if total_anom > 0:
                    st.error(f"🚨 TERDETEKSI {total_anom} data anomali. Contoh pasien: {', '.join(map(str,siren_names[:6]))}")

                    # 1) show dramatic popup
                    st.markdown("""<script>showDramaticWarning("ANOMALI TENSI TERDETEKSI!");</script>""", unsafe_allow_html=True)

                    # 2) prepare siren audio and wait for dramatic popup countdown
                    wav = generate_siren_wav(duration=2.0)
                    datauri = wav_bytes_to_datauri(wav)
                    time.sleep(2.1)

                    # 3) show full siren overlay with audio
                    render_siren_overlay(datauri, title_text="PERINGATAN: Anomali Tensi Terdeteksi!")

                    st.subheader("Detail pasien bermasalah")
                    for p in siren_names:
                        with st.expander(f"Pasien: {p}"):
                            sub = result[result['Nama']==p].reset_index(drop=True)
                            st.dataframe(sub)
                            fig, ax = plt.subplots(figsize=(8,3))
                            ax.plot(sub['Tanggal'], sub['Systolic'], marker='o', label='Systolic')
                            ax.plot(sub['Tanggal'], sub['Diastolic'], marker='o', label='Diastolic')
                            # highlight anomalies if present
                            if 'Anom_Total' in sub.columns and sub['Anom_Total'].any():
                                anom_dates = sub[sub['Anom_Total']]['Tanggal']
                                anom_vals = sub[sub['Anom_Total']]['Systolic']
                                ax.scatter(anom_dates, anom_vals, color='red', marker='x', s=90, label='Anomali')
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

    if st.button("⬅ Kembali ke Beranda"):
        st.session_state.page = "beranda"
    st.stop()

# -----------------------
# HALAMAN: PERSONAL (dengan petunjuk di kanan)
# -----------------------
if st.session_state.page == "personal":
    st.markdown("<div style='display:flex;align-items:flex-start;gap:18px;'>", unsafe_allow_html=True)
    # left: main form
    st.markdown("<div style='flex:2;'>", unsafe_allow_html=True)
    st.header("👤 Analisis Personal")
    st.write("Masukkan nama, pilih jumlah data (max 10), isi pasangan Systolic & Diastolic, lalu klik Analisis (RK4).")
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
    st.markdown("</div>", unsafe_allow_html=True)

    # right: petunjuk (glass card)
    st.markdown("<div style='flex:1;'>", unsafe_allow_html=True)
    st.markdown('<div class="glass"><h4 style="color:#0b63d9;margin-bottom:6px;">Petunjuk</h4><p style="color:var(--muted);margin:0;">Masukkan data dalam urutan kronologis (yang paling awal dahulu). Prediksi akan dilakukan 1 langkah ke depan berdasarkan dua data terakhir menggunakan RK4 sederhana.</p><hr style="border:none;border-top:1px solid rgba(11,99,217,0.06);margin:12px 0;"><p style="margin:0;color:var(--muted);font-size:13px;"><b>Catatan:</b> Hasil komputasi bersifat ilustrasi. Untuk keputusan medis, konsultasi tenaga kesehatan diperlukan.</p></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

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
            if bool(last.get('Anom_Total', False)):
                st.error("⚠️ Terdeteksi anomali pada data terakhir!")
                # dramatic popup + siren overlay
                st.markdown("""<script>showDramaticWarning("ANOMALI TERDETEKSI PADA DATA!");</script>""", unsafe_allow_html=True)
                wav = generate_siren_wav(duration=2.0)
                datauri = wav_bytes_to_datauri(wav)
                time.sleep(2.1)
                render_siren_overlay(datauri, title_text="PERINGATAN: Anomali pada Data!")
            else:
                st.success("✔ Data terakhir tampak normal.")

            # show predictions
            if pred_s is not None:
                st.markdown(f"**Prediksi RK4 (1 langkah ke depan)** — Sistolik: **{pred_s:.2f}**, Diastolik: **{pred_d:.2f}**")

    if st.button("⬅ Kembali ke Beranda"):
        st.session_state.page = "beranda"
    st.stop()

# -----------------------
# HALAMAN: HASIL ANALISIS
# -----------------------
if st.session_state.page == "hasil":
    st.header("Hasil Analisis Terakhir")
    if st.session_state.last_result is None:
        st.info("Belum ada hasil analisis. Lakukan analisis pada halaman Input Data atau Personal terlebih dahulu.")
    else:
        st.write("Context:", st.session_state.last_context)
        st.dataframe(st.session_state.last_result)
        df_show = st.session_state.last_result
        total_anom = int(df_show['Anom_Total'].sum()) if 'Anom_Total' in df_show.columns else 0
        st.markdown(f"**Total anomali terdeteksi:** {total_anom}")

    if st.button("⬅ Kembali ke Beranda"):
        st.session_state.page = "beranda"
    st.stop()

# -----------------------
# HALAMAN: MENGAPA RK4?
# -----------------------
if st.session_state.page == "rk4info":
    st.header("Mengapa Runge–Kutta Orde-4 (RK4)?")
    st.write("Ringkasan singkat kenapa memakai RK4 untuk prediksi sederhana:")
    st.markdown("""
    - **RK4** adalah metode numerik berorde tinggi yang sering dipakai untuk menyelesaikan persamaan diferensial biasa (ODE).
    - Untuk data time-series seperti tekanan darah, RK4 memberikan estimasi yang lebih stabil dibanding metode sederhana (mis. Euler) karena mengkombinasikan beberapa evaluasi turunan (k1..k4).
    - Dalam aplikasi demo ini, kita mengestimasi turunan (perubahan) dari dua titik terakhir dan menjalankan satu langkah RK4 — hasilnya seringkali *lebih halus* dan kurang berisik daripada pendekatan 1-step sederhana.
    - **Catatan klinis:** prediksi ini bersifat ilustrasi. Untuk keputusan medis, gunakan model yang divalidasi klinis dan data kontekstual (obat, riwayat, aktivitas, dsb.).
    """)
    st.markdown("**Bagan singkat (intuisi):**")
    st.image("https://upload.wikimedia.org/wikipedia/commons/1/1a/Runge-Kutta_method.svg", caption="Sketsa Runge-Kutta (ilustrasi)")

    if st.button("⬅ Kembali ke Beranda"):
        st.session_state.page = "beranda"
    st.stop()

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.markdown('<div style="color:var(--muted);">Catatan: Aplikasi ini untuk tujuan demo/pendukung edukasi — bukan pengganti konsultasi medis. Sesuaikan aturan deteksi dan threshold jika diperlukan.</div>', unsafe_allow_html=True)
