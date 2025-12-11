# app.py — NADI (RK4) — FINAL 
# ==============================================================================

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
import json

# -----------------------
# Config / State bootstrap
# -----------------------
st.set_page_config(page_title="NADI (RK4) — Soft Blue", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "beranda"
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_context" not in st.session_state:
    st.session_state.last_context = None
if "greet_shown" not in st.session_state:
    st.session_state.greet_shown = False
if "nadibot_min" not in st.session_state:
    st.session_state.nadibot_min = False

# stats file (visitor + analyses)
STATS_PATH = "nadi_stats.json"

def load_stats():
    default = {"visitors": 0, "analyses": 0}
    try:
        if os.path.exists(STATS_PATH):
            with open(STATS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return default
    except Exception:
        return default

def save_stats(stats):
    try:
        with open(STATS_PATH, "w", encoding="utf-8") as f:
            json.dump(stats, f)
    except Exception:
        pass

# increment visitor once when first load app in session
if not st.session_state.get("visitor_counted"):
    stats = load_stats()
    stats["visitors"] = stats.get("visitors", 0) + 1
    save_stats(stats)
    st.session_state.visitor_counted = True

# -----------------------
# GLOBAL CSS + MOVING PATTERN + nadibot styles
# -----------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    :root{ --bg1: #f3f9ff; --bg2: #eaf6ff; --primary-1: #0b63d9; --muted: #6b7280; }

    html, body, [class*="css"] { font-family: Inter, "Times New Roman", serif; background: linear-gradient(180deg, var(--bg1), var(--bg2)); }

    .big-nadi-title { font-family: "Times New Roman", serif; font-size:56px; font-weight:800; color:white !important; text-shadow:0 0 14px rgba(0,0,0,0.35); text-align:center; margin-top:6px; margin-bottom:8px; }
    .nadi-desc { text-align:center; color:var(--muted); max-width:920px; margin:auto; margin-bottom:14px; font-size:15px; line-height:1.45; }
    .glass { background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(246,251,255,0.9)); border-radius:12px; padding:14px; box-shadow:0 8px 28px rgba(11,99,217,0.06); border:1px solid rgba(255,255,255,0.6); transition: transform .14s ease, box-shadow .14s ease; }
    .glass:hover { transform: translateY(-6px); box-shadow:0 18px 40px rgba(11,99,217,0.08); }
    .spacer { margin-top:12px; margin-bottom:12px; }

    /* MOVING PATTERN - low z so popups are above */
    body::before {
        content: "";
        position: fixed;
        inset: 0;
        z-index: 0;
        opacity: 0.09;
        background-image:
            url("https://i.ibb.co/8MmS2Xh/stetho.png"),
            url("https://i.ibb.co/1TcfPjT/heartpulse.png"),
            url("https://i.ibb.co/W2qQwxp/tensi.png");
        background-repeat: repeat;
        background-size: 180px, 150px, 170px;
        filter: blur(0.4px);
        animation: drift 60s linear infinite;
        pointer-events: none;
    }
    @keyframes drift {
        0% { background-position: 0px 0px, 0px 0px, 0px 0px; }
        50% { background-position: 200px 150px, 150px 200px, 250px 180px; }
        100% { background-position: 0px 0px, 0px 0px, 0px 0px; }
    }

    /* NadiBot (chibi) - bottom right */
    #nadibot {
        position: fixed;
        right: 18px;
        bottom: 18px;
        z-index: 10000000000;
        width: 86px;
        height: 86px;
        border-radius: 18px;
        box-shadow: 0 12px 30px rgba(0,0,0,0.18);
        background: linear-gradient(135deg,#ffffff,#e6fbf5);
        display:flex; align-items:center; justify-content:center;
        cursor:pointer;
    }
    #nadibot img { width:72px; height:72px; object-fit:contain; border-radius:12px; }
    #nadibot-bubble {
        position: fixed;
        right: 116px;
        bottom: 36px;
        z-index: 10000000000;
        background: linear-gradient(135deg,#ffffff,#f0fff6);
        color: #0b63d9;
        padding:10px 14px;
        border-radius:14px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        font-weight:600;
        display:none;
    }
    #nadibot-min {
        position: fixed;
        right: 18px;
        bottom: 110px;
        z-index: 10000000000;
        display:none;
    }

    /* small responsive */
    @media (max-width:600px) {
        #nadibot { width:70px; height:70px; right:12px; bottom:12px; }
        #nadibot-bubble { right:86px; bottom:28px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# Audio helpers
# -----------------------
def generate_siren_wav(duration=3.0, sr=44100):
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    mod = 0.5 * (1 + np.sin(2 * np.pi * 2.2 * t))
    freq = 700 + 600 * np.sin(2 * np.pi * 1.0 * t)
    tone = 0.9 * np.sin(2 * np.pi * freq * t) * (0.6 + 0.4 * mod)
    env = np.linspace(1, 0.001, len(t))
    tone = tone * env
    tone = np.clip(tone, -1, 1)
    buf = BytesIO()
    sf.write(buf, tone, sr, format='WAV')
    buf.seek(0)
    return buf.read()

def generate_ting_wav(duration=0.45, sr=44100):
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
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
# RK4 helpers & anomaly detection (same logic)
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
# Popups (components.html so fixed overlay works reliably)
# -----------------------
def render_normal_popup(duration_ms=2000):
    wav = generate_ting_wav(duration=0.45)
    datauri = wav_bytes_to_datauri(wav)
    html = f"""
    <html><head><meta charset="utf-8"/><style>
    @keyframes pop{{0%{{transform:scale(.4);opacity:0}}60%{{transform:scale(1.18);opacity:1}}100%{{transform:scale(1);opacity:1}}}}
    @keyframes fadein{{0%{{opacity:0}}100%{{opacity:1}}}}
    @keyframes pulse{{0%{{transform:scale(1)}}50%{{transform:scale(1.15)}}100%{{transform:scale(1)}}}}
    @keyframes spark{{0%{{opacity:0;transform:scale(.5)}}25%{{opacity:1;transform:scale(1.2) translateY(-8px)}}70%{{opacity:.4;transform:scale(1)}}100%{{opacity:0;transform:scale(.5)}}}}
    </style></head><body>
    <div id="normal-popup-root" style="position:fixed; inset:0; background:rgba(0,0,0,0.45); backdrop-filter:blur(7px); z-index:999999999; display:flex; justify-content:center; align-items:center; animation:fadein .20s forwards;">
      <div style="width:520px; background:linear-gradient(145deg,#00e09f,#00b46f); border-radius:30px; padding:32px; text-align:center; color:white; box-shadow:0 40px 80px rgba(0,50,20,0.35); animation:pop .35s cubic-bezier(.2,.9,.2,1); position:relative;">
        <div style="font-size:90px; margin-bottom:10px; animation:pulse 1.4s infinite; filter:drop-shadow(0 0 20px rgba(0,255,180,0.8));">✔️</div>
        <h2 style="margin:0; font-size:36px; font-weight:900;">Datamu Normal!</h2>
        <p style="font-size:20px; opacity:0.95; margin-top:8px;">Jaga kesehatan yaaa!! 💚✨</p>
        <div style="position:absolute; top:-10px; left:70px; width:14px; height:14px; background:white; border-radius:50%; opacity:0; box-shadow:0 0 14px white; animation:spark 1.5s infinite;"></div>
        <div style="position:absolute; top:-20px; right:80px; width:14px; height:14px; background:white; border-radius:50%; opacity:0; box-shadow:0 0 14px white; animation:spark 1.5s infinite .3s;"></div>
        <div style="position:absolute; bottom:-10px; left:85px; width:14px; height:14px; background:white; border-radius:50%; opacity:0; box-shadow:0 0 14px white; animation:spark 1.5s infinite .6s;"></div>
      </div>
      <audio autoplay><source src="{datauri}" type="audio/wav"></audio>
    </div>
    <script>setTimeout(function(){{var el=document.getElementById('normal-popup-root'); if(el && el.parentNode) el.parentNode.removeChild(el);}}, {duration_ms});</script>
    </body></html>
    """
    components.html(html, height=0, width=0)

def render_warning_inline(duration_ms=1000, siren_duration=3.0):
    wav = generate_siren_wav(duration=siren_duration)
    datauri = wav_bytes_to_datauri(wav)
    html = f"""
    <html><head><meta charset="utf-8"/><style>
    @keyframes fadeIn{{from{{opacity:0}}to{{opacity:1}}}}
    @keyframes popWarn{{0%{{transform:scale(.28);opacity:0}}50%{{transform:scale(1.18);opacity:1}}100%{{transform:scale(1);opacity:1}}}}
    @keyframes shakeWarn{{0%{{transform:translateX(0)}}25%{{transform:translateX(-18px)}}50%{{transform:translateX(14px)}}75%{{transform:translateX(-10px)}}100%{{transform:translateX(0)}}}}
    @keyframes glowWarn{{from{{filter:drop-shadow(0 0 18px rgba(255,80,80,0.85))}}to{{filter:drop-shadow(0 0 30px rgba(255,0,0,1))}}}}
    </style></head><body>
    <div id="warn-root" style="position:fixed; inset:0; background:rgba(0,0,0,0.88); backdrop-filter:blur(10px); z-index:999999999; display:flex; justify-content:center; align-items:center; animation:fadeIn .12s ease-out;">
      <div style="width:620px; max-width:92%; background:linear-gradient(135deg,#ff2d2d,#8b0000); border-radius:30px; padding:40px 32px; text-align:center; color:white; box-shadow:0 40px 120px rgba(255,0,0,0.45); animation:popWarn .9s cubic-bezier(.18,.89,.32,1.28); position:relative;">
        <div style="font-size:108px; margin-bottom:10px; filter:drop-shadow(0 0 28px rgba(255,0,0,1)); animation:glowWarn .4s infinite alternate;">🚨</div>
        <h1 style="margin:0; font-size:40px; font-weight:900; letter-spacing:0.6px;">PERINGATAN TENSI TIDAK NORMAL!</h1>
        <p style="font-size:20px; opacity:.95; margin-top:8px;">Hipertensi / hipotensi terdeteksi. Mohon cek ulang datamu.</p>
      </div>
      <audio autoplay><source src="{datauri}" type="audio/wav"></audio>
    </div>
    <script>setTimeout(function(){{var el=document.getElementById('warn-root'); if(el && el.parentNode) el.parentNode.removeChild(el);}}, {duration_ms});</script>
    </body></html>
    """
    components.html(html, height=0, width=0)

# -----------------------
# Greeting popup (NadiBot chibi) - shows once per session when on beranda
# -----------------------
def render_greeting_nadibot():
    # simple chibi SVG (inline) — cute, small, fast
    # appears from bottom, waves, then disappears (1.0s)
    html = """
    <html><head><meta charset="utf-8"/><style>
    @keyframes popIn {0%{transform:translateY(120%);opacity:0}60%{transform:translateY(-6%);opacity:1}100%{transform:translateY(0);opacity:1}}
    @keyframes wave {0%{transform:rotate(0deg)}50%{transform:rotate(12deg)}100%{transform:rotate(0deg)}}
    </style></head><body>
    <div id="greet-root" style="position:fixed; inset:0; display:flex; justify-content:center; align-items:flex-end; z-index:9999999999; pointer-events:none;">
      <div style="pointer-events:auto; margin-bottom:36px; background:linear-gradient(180deg,#fff,#f7fffb); padding:12px 16px; border-radius:16px; box-shadow:0 18px 50px rgba(0,0,0,0.18); display:flex; align-items:center; gap:12px; animation:popIn .9s cubic-bezier(.2,.9,.2,1) forwards;">
        <!-- chibi inline SVG -->
        <div style="width:86px; height:86px; display:flex; align-items:center; justify-content:center;">
          <svg width="70" height="70" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <linearGradient id="g1" x1="0" x2="1" y1="0" y2="1">
                <stop offset="0" stop-color="#ffd6e0"/><stop offset="1" stop-color="#ffb3d0"/>
              </linearGradient>
            </defs>
            <rect rx="18" width="100" height="100" fill="url(#g1)"/>
            <g transform="translate(12,18)">
              <circle cx="30" cy="20" r="14" fill="#fff"/>
              <circle cx="21" cy="17" r="3.5" fill="#1f2937"/>
              <circle cx="39" cy="17" r="3.5" fill="#1f2937"/>
              <path d="M18 28 q12 10 24 0" stroke="#1f2937" stroke-width="2.5" fill="none" stroke-linecap="round"/>
            </g>
          </svg>
        </div>
        <div style="color:#0b63d9; font-weight:700; font-size:18px;">
          <div>Haii! Aku NadiBot 👋</div>
          <div style="font-weight:500; font-size:14px; color:#374151;">Mau cek tensi hari ini? Klik Personal atau Upload data.</div>
        </div>
      </div>
    </div>
    <script>
    setTimeout(function(){var e=document.getElementById('greet-root'); if(e && e.parentNode) e.parentNode.removeChild(e);}, 1200);
    </script>
    </body></html>
    """
    components.html(html, height=0, width=0)

# -----------------------
# Loading overlay (used before running heavy analysis) - shows heartbeat animation
# -----------------------
def render_loading_overlay(duration_ms=1200):
    html = """
    <html><head><meta charset="utf-8"/><style>
    @keyframes beat {0%{transform:scale(1)}25%{transform:scale(1.18)}50%{transform:scale(1)}75%{transform:scale(1.08)}100%{transform:scale(1)}}
    </style></head><body>
    <div style="position:fixed; inset:0; background:rgba(255,255,255,0.0); z-index:9999999998; display:flex; align-items:center; justify-content:center; pointer-events:none;">
      <div style="width:220px; height:120px; border-radius:12px; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:10px;">
        <div style="font-size:34px; transform-origin:center; animation:beat 1.2s infinite;">💓</div>
        <div style="font-weight:700; color:#0b63d9;">NADI sedang membaca tensi...</div>
      </div>
    </div>
    <script>setTimeout(function(){document.body.innerHTML = document.body.innerHTML;}, """ + str(duration_ms) + """);</script>
    </body></html>
    """
    # show via components.html for reliable overlay
    components.html(html, height=0, width=0)

# ============================================================
# BERANDA
# ============================================================
if st.session_state.page == "beranda":
    # greeting popup once per session
    if not st.session_state.greet_shown:
        render_greeting_nadibot()
        st.session_state.greet_shown = True

    st.markdown("<div class='big-nadi-title'>❤️ NADI : Numeric Analysis of Diastolic & Systolic</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='nadi-desc'><b>Adalah ruang sederhana untuk membaca alur tekanan darah Anda melalui pendekatan komputasi.</b><br>"
        "Dengan memanfaatkan metode <b>RK4</b> dan proses pengkodingan yang turut terbantu oleh kecerdasan buatan, <b>NADI</b> menghadirkan analisis yang ringan, intuitif, dan mudah dipahami.<br><br>"
        "<b>NADI bukan alat diagnosis medis</b>. Hasil yang ditampilkan hanya gambaran komputasi, bukan pengganti konsultasi tenaga kesehatan profesional.<br><br><i>Selamat datang. Biarkan NADI membaca aliran kesehatan Anda.</i></div>",
        unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='glass'><h3 style='color:var(--primary-1)'>Input Data (Puskesmas / Klinik)</h3><p style='color:var(--muted)'>Upload CSV/XLSX berisi banyak pasien. Sistem akan memproses tiap pasien, mendeteksi hipertensi/hipotensi, dan memprediksi nilai berikutnya.</p></div>", unsafe_allow_html=True)
        if st.button("➡ Masuk ke Input Data"):
            st.session_state.page = "input"
    with c2:
        st.markdown("<div class='glass'><h3 style='color:var(--primary-1)'>Analisis Personal</h3><p style='color:var(--muted)'>Masukkan 1–10 data tensi</p></div>", unsafe_allow_html=True)
        if st.button("➡ Masuk ke Personal"):
            st.session_state.page = "personal"

    # beranda nav buttons (clean)
    st.markdown("<br>", unsafe_allow_html=True)
    nav1, nav2, nav3 = st.columns(3)
    with nav1:
        if st.button("📊 Hasil Analisis Terakhir"):
            st.session_state.page = "hasil"
    with nav2:
        if st.button("❔ Mengapa RK4?"):
            st.session_state.page = "rk4info"
    with nav3:
        if st.button("🔄 Reset"):
            st.session_state.last_result = None
            st.session_state.last_context = None
            st.success("Riwayat berhasil direset!")

    # visitor / analyses stats
    stats = load_stats()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("👥 Total Pengunjung (sesi)", stats.get("visitors", 0))
    with col2:
        st.metric("📊 Total Analisis", stats.get("analyses", 0))

    # optional debug test buttons
    dbg1, dbg2 = st.columns([1,1])
    with dbg1:
        if st.button("🔔 Test Warning (dev)"):
            render_warning_inline(duration_ms=1000, siren_duration=3.0)
    with dbg2:
        if st.button("✅ Test Normal (dev)"):
            render_normal_popup(duration_ms=1800)

    st.markdown("---")
    st.subheader("Contoh Template Data (Download)")
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

    # NadiBot markup (minimize toggle handled by JS)
    st.markdown(
        """
        <div id="nadibot" title="NadiBot (klik untuk pesan)">
            <img src="https://i.ibb.co/2t5vVdP/chibi-nadi.png" alt="NadiBot"/>
        </div>
        <div id="nadibot-bubble">Haii! Mau cek tensi?</div>
        <script>
        const bot = document.getElementById('nadibot');
        const bubble = document.getElementById('nadibot-bubble');
        let minimized = false;
        bot.onclick = function(e){
            minimized = !minimized;
            if(!minimized){
                bubble.style.display = 'block';
                setTimeout(()=>bubble.style.display='none',2200);
            } else {
                bubble.style.display = 'none';
            }
        }
        </script>
        """,
        unsafe_allow_html=True,
    )

    st.stop()

# ============================================================
# INPUT DATA (UPLOAD) PAGE
# ============================================================
if st.session_state.page == "input":
    st.header("📁 Analisis Data Populasi (Upload CSV / XLSX)")
    uploaded = st.file_uploader("Upload CSV / XLSX (minimal kolom: Nama, Systolic, Diastolic)", type=["csv","xlsx"])
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
        st.info("Preview data (50 baris pertama):")
        st.dataframe(df.head(50))

        required = {"Nama","Systolic","Diastolic"}
        if not required.issubset(df.columns):
            st.error(f"Kolom minimal harus ada: {sorted(required)}")
            st.stop()

        # handle tanggal
        if "Tanggal" in df.columns:
            df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors='coerce')
            if df["Tanggal"].isna().any():
                df["Tanggal"] = df["Tanggal"].fillna(method="ffill")
        else:
            today = datetime.now().date()
            n = len(df)
            df["Tanggal"] = [pd.Timestamp(today - timedelta(days=(n-1-i))) for i in range(n)]

        if run:
            # show loading overlay (small delay so user sees animation)
            render_loading_overlay(duration_ms=1200)
            time.sleep(1.2)

            parts = []
            alert_needed = False
            alert_names = []

            for name, g in df.groupby("Nama", sort=False):
                g2 = g.sort_values("Tanggal").reset_index(drop=True)
                g2 = g2[["Nama","Tanggal","Systolic","Diastolic"]].copy()
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

            result = pd.concat(parts, ignore_index=True)
            st.subheader("Hasil Analisis")
            st.dataframe(result)
            st.session_state.last_result = result
            st.session_state.last_context = {"mode":"Input","file":uploaded.name}

            # increment analyses counter
            stats = load_stats()
            stats["analyses"] = stats.get("analyses", 0) + 1
            save_stats(stats)

            if alert_needed:
                st.error(f"🚨 Anomali terdeteksi pada: {', '.join(alert_names[:8])}")
                render_warning_inline(duration_ms=1000, siren_duration=3.0)
            else:
                render_normal_popup(duration_ms=2000)
                st.success("✔ Tidak ada hipertensi/hipotensi terdeteksi.")

    if st.button("⬅ Kembali"):
        st.session_state.page = "beranda"
    st.stop()

# ============================================================
# PERSONAL ANALYSIS PAGE
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

        # loading animation
        render_loading_overlay(duration_ms=1200)
        time.sleep(1.2)

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

        # increment analyses counter
        stats = load_stats()
        stats["analyses"] = stats.get("analyses", 0) + 1
        save_stats(stats)

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
            render_warning_inline(duration_ms=1000, siren_duration=3.0)
        else:
            render_normal_popup(duration_ms=2000)
            st.success("✔ Datamu Normal. Jaga Kesehatan Yaa!!!")

        if pred_s is not None:
            st.markdown(f"**Prediksi RK4 (1 langkah)** — Sistolik: **{pred_s:.2f}**, Diastolik: **{pred_d:.2f}**")

        st.session_state.last_result = dfp
        st.session_state.last_context = {"mode":"Personal", "name":name}

    if st.button("⬅ Kembali"):
        st.session_state.page = "beranda"
    st.stop()

# ============================================================
# RESULTS PAGE
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
stats = load_stats()
st.markdown(f"<div style='color:var(--muted); font-size:13px;'>NADI | RK4 — aplikasi edukasi numerik. Visitors: {stats.get('visitors',0)} • Analyses: {stats.get('analyses',0)}</div>", unsafe_allow_html=True)
