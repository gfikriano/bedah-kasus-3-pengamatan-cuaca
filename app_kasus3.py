"""
Dashboard Streamlit - Bedah Kasus 3: Dataset Pengamatan Cuaca
Nama : Ganesia Alkhan
NIM  : 241051019

Jalankan:  streamlit run app_kasus3.py
Data    :  data_pengamatan_cuaca.csv (satu folder dengan file ini). Jika tidak ada,
           data sintetis 1.000 baris dibuat otomatis dengan seed yang sama seperti di notebook.
"""
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------- Identitas & konstanta
NAMA, NIM = "Ganesia Alkhan", "241051019"
BIRU, ORANYE, ORANYE_TUA = "#2E9BD6", "#F5A03A", "#D9731A"
BLUES = ["#08306B", "#08519C", "#2171B5", "#4292C6", "#6BAED6", "#9ECAE1", "#C6DBEF", "#DEEBF7"]
SUHU, RH = "Suhu (°C)", "Kelembaban (%)"
KONDISI = ["Cerah", "Cerah Berawan", "Berawan", "Hujan", "Hujan Petir"]      # urutan tata letak, bukan ordinal
WARNA_KONDISI = {"Cerah": "#F6B93B", "Cerah Berawan": "#A8D08D", "Berawan": "#9AA5B1", "Hujan": "#3C8DBC", "Hujan Petir": "#4B3F72"}
KOLOM_WAJIB = ["Kota", SUHU, RH, "Kondisi"]

st.set_page_config(page_title=f"Dashboard Cuaca - {NIM}", page_icon="🌦️", layout="wide")

def buat_data_cuaca(n=1000, seed=241051019):
    """Dataset pengamatan cuaca sintetis. 1 baris = 1 pengamatan di 1 kota.
    Kolom: Kota (nominal), Suhu (°C) (interval), Kelembaban (%) (rasio), Kondisi (nominal).
    """
    rng = np.random.default_rng(seed)

    # kota: (bobot pengamatan, suhu rata-rata harian °C, amplitudo harian °C, RH dasar %, kecenderungan hujan)
    kota = {
        "Jakarta":    (0.20, 28.5, 3.5, 74, 0.30),
        "Bandung":    (0.14, 23.0, 3.8, 80, 0.40),   # dataran tinggi: lebih sejuk dan lembap
        "Surabaya":   (0.14, 29.0, 3.5, 72, 0.25),
        "Medan":      (0.12, 28.0, 3.3, 77, 0.38),
        "Semarang":   (0.11, 28.8, 3.5, 74, 0.28),
        "Yogyakarta": (0.11, 27.5, 3.8, 74, 0.30),
        "Makassar":   (0.09, 28.7, 3.2, 75, 0.27),
        "Denpasar":   (0.09, 27.8, 3.0, 77, 0.26),
    }
    nama = list(kota)
    bobot = np.array([v[0] for v in kota.values()])
    idx = rng.choice(len(nama), size=n, p=bobot / bobot.sum())
    t0, amp, rh0, hujan = (np.array([v[i] for v in kota.values()])[idx] for i in (1, 2, 3, 4))

    # Jam pengamatan (tidak disimpan) menentukan siklus harian suhu dan peluang hujan
    jam = rng.integers(0, 24, n)
    suhu_dasar = t0 + amp * np.cos(2 * np.pi * (jam - 15) / 24)          # puncak sekitar pukul 15.00
    siang = (jam >= 7) & (jam <= 17)

    kondisi = np.empty(n, dtype=object)
    for i in range(n):
        sore = 13 <= jam[i] <= 18
        p_hujan = min(0.70, hujan[i] * (1.7 if sore else (0.6 if 7 <= jam[i] <= 11 else 0.8)))
        p_petir = p_hujan * (0.25 if sore else 0.10)
        sisa = 1 - p_hujan
        w = np.array([0.45, 0.33, 0.22]) if siang[i] else np.array([0.25, 0.35, 0.40])
        p = np.concatenate([w * sisa, [p_hujan - p_petir, p_petir]])
        kondisi[i] = rng.choice(["Cerah", "Cerah Berawan", "Berawan", "Hujan", "Hujan Petir"], p=p / p.sum())

    # Pengaruh kondisi: hujan menurunkan suhu dan menaikkan kelembaban
    efek = {"Cerah": (1.2, -7), "Cerah Berawan": (0.0, 0), "Berawan": (-0.8, 4),
            "Hujan": (-3.0, 12), "Hujan Petir": (-4.0, 14)}
    dT = np.array([efek[k][0] for k in kondisi]) + rng.normal(0, 0.6, n)
    dH = np.array([efek[k][1] for k in kondisi])

    suhu = suhu_dasar + dT + rng.normal(0, 1.0, n)
    rh = rh0 - 3.0 * (suhu_dasar - t0) + dH + rng.normal(0, 4.0, n)      # suhu naik -> RH turun
    rh = np.where(rh > 100, 200 - rh, rh)                                  # pantulkan di batas 100%
    rh = np.clip(np.round(rh), 30, 100).astype(int)
    suhu = np.round(np.clip(suhu, 12, 38), 1)

    return pd.DataFrame({"Kota": np.array(nama)[idx], "Suhu (°C)": suhu,
                         "Kelembaban (%)": rh, "Kondisi": kondisi})


# ----------------------------------------------------------------------------- Gaya
st.markdown("""
<style>
.block-container { padding-top: 1.6rem; padding-bottom: 4.5rem; }
.hdr { display:flex; align-items:stretch; gap:16px; margin-bottom:.4rem; }
.hdr .stripe { display:flex; flex-direction:column; width:8px; border-radius:4px; overflow:hidden; }
.hdr .stripe span { flex:1; }
.hdr h1 { margin:0; padding:0; font-size:2rem; line-height:1.15; }
.hdr p { margin:.25rem 0 0 0; color:#5b6068; }
.hdr .idbox { margin-left:auto; text-align:right; padding:.5rem .9rem; border:1px solid #d9dce1; border-radius:8px; font-size:.92rem; line-height:1.45; }
.tag { border-left:5px solid var(--c); padding:.35rem .8rem; margin:.2rem 0 .8rem 0; background:rgba(127,127,127,.08); border-radius:0 6px 6px 0; font-size:.93rem; }
.footer-id { position:fixed; left:0; right:0; bottom:0; z-index:999; text-align:center; font-size:.82rem;
             padding:.4rem; background:rgba(255,255,255,.96); color:#444; border-top:1px solid #d9dce1; }
.sid { border:1px solid #d9dce1; border-radius:8px; padding:.6rem .8rem; line-height:1.5; font-size:.92rem; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- Utilitas
def id_num(x, d=0):
    """Angka gaya Indonesia: 1.234,56"""
    return f"{x:,.{d}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def tag(teks, warna):
    st.markdown(f'<div class="tag" style="--c:{warna}">{teks}</div>', unsafe_allow_html=True)


def rapikan(fig, tinggi=420, judul=None):
    fig.update_layout(template="plotly_white", height=tinggi, margin=dict(l=10, r=10, t=55 if judul else 25, b=10),
                      legend_title_text="", title=dict(text=judul, font=dict(size=15)) if judul else None)
    return fig


def warna_grup(gcol, utama):
    """Argumen warna plotly menurut kelompok (Kondisi memakai warna semantik)."""
    if gcol == "Kondisi":
        return dict(color="Kondisi", color_discrete_map=WARNA_KONDISI, category_orders={"Kondisi": KONDISI})
    if gcol == "Kota":
        return dict(color="Kota", color_discrete_sequence=px.colors.qualitative.T10)
    return dict(color_discrete_sequence=[utama])


def qq_fig(serie, warna):
    """QQ-plot manual terhadap distribusi normal (tanpa scipy)."""
    ys = np.sort(serie.to_numpy())
    nd = NormalDist()
    teori = [nd.inv_cdf((i - 0.5) / len(ys)) for i in range(1, len(ys) + 1)]
    mu, sd = ys.mean(), (ys.std(ddof=1) if len(ys) > 1 else 0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=teori, y=ys, mode="markers", marker=dict(color=warna, size=5, opacity=.7), name="Data"))
    fig.add_trace(go.Scatter(x=[min(teori), max(teori)], y=[mu + sd * min(teori), mu + sd * max(teori)], mode="lines",
                             line=dict(color="#333", dash="dash"), name="Garis normal"))
    fig.update_xaxes(title="Kuantil normal teoretis")
    return fig


# ----------------------------------------------------------------------------- Data
@st.cache_data(show_spinner=False)
def muat_bawaan():
    csv = Path(__file__).with_name("data_pengamatan_cuaca.csv")
    return pd.read_csv(csv) if csv.exists() else buat_data_cuaca(1000)


def siapkan(df):
    df = df.copy()
    df["Kota"], df["Kondisi"] = df["Kota"].astype(str), df["Kondisi"].astype(str)
    df[SUHU], df[RH] = df[SUHU].astype(float), df[RH].astype(float)
    df["Hujan"] = df["Kondisi"].isin(["Hujan", "Hujan Petir"])
    return df


# ----------------------------------------------------------------------------- Sidebar
with st.sidebar:
    st.markdown(f'<div class="sid"><b>{NAMA}</b><br>NIM {NIM}</div>', unsafe_allow_html=True)
    st.markdown("### Filter data")
    data_mentah = muat_bawaan()
    with st.expander("Gunakan data lain (CSV)"):
        up = st.file_uploader("Unggah CSV dengan kolom: Kota, Suhu (°C), Kelembaban (%), Kondisi", type="csv")
        if up is not None:
            try:
                tmp = pd.read_csv(up)
                hilang = [k for k in KOLOM_WAJIB if k not in tmp.columns]
                if hilang:
                    st.error(f"Kolom belum lengkap, tidak ada: {', '.join(hilang)}. Memakai data bawaan.")
                elif not tmp[RH].between(0, 100).all():
                    st.error("Kelembaban harus berada pada 0-100%. Memakai data bawaan.")
                else:
                    data_mentah = tmp
                    st.success(f"{len(tmp):,} baris dimuat.".replace(",", "."))
            except Exception as e:  # noqa: BLE001
                st.error(f"CSV tidak bisa dibaca ({e}). Memakai data bawaan.")

df = siapkan(data_mentah)

semua_kota = sorted(df["Kota"].unique())
kota_pilih = st.sidebar.multiselect("Kota", semua_kota, default=semua_kota)
semua_kon = [k for k in KONDISI if k in set(df["Kondisi"])] + sorted(set(df["Kondisi"]) - set(KONDISI))
kon_pilih = st.sidebar.multiselect("Kondisi", semua_kon, default=semua_kon)
smin, smax = float(np.floor(df[SUHU].min())), float(np.ceil(df[SUHU].max()))
suhu_rng = st.sidebar.slider("Suhu (°C)", smin, smax, (smin, smax), step=0.5, format="%.1f")
hmin, hmax = int(df[RH].min()), int(df[RH].max())
rh_rng = st.sidebar.slider("Kelembaban (%)", hmin, hmax, (hmin, hmax))

st.sidebar.markdown("### Pengaturan tampilan")
SATUAN = st.sidebar.radio("Satuan suhu pada grafik", ["°C", "K", "°F"], horizontal=True,
                          help="Mengganti satuan hanya mengubah tampilan. Filter di atas tetap dalam °C.")
TS = f"Suhu ({SATUAN})"


def konversi(c):
    return c if SATUAN == "°C" else (c + 273.15 if SATUAN == "K" else c * 9 / 5 + 32)


mask = (df["Kota"].isin(kota_pilih) & df["Kondisi"].isin(kon_pilih) & df[SUHU].between(*suhu_rng) & df[RH].between(*rh_rng))
f = df[mask].copy()
f[TS] = konversi(f[SUHU])

# ----------------------------------------------------------------------------- Header
st.markdown(f"""
<div class="hdr">
  <div class="stripe"><span style="background:{BIRU}"></span><span style="background:{ORANYE}"></span></div>
  <div><h1>Dashboard Pengamatan Cuaca</h1>
       <p>Bedah Kasus 3. Data sintetis 1.000 pengamatan di 8 kota, divisualisasikan sesuai skala pengukuran tiap kolom.</p></div>
  <div class="idbox"><b>{NAMA}</b><br>NIM {NIM}</div>
</div>
""", unsafe_allow_html=True)
st.caption("Kode warna mengikuti gambar tugas: biru untuk data nominal (Kota, Kondisi), oranye untuk numerik (Suhu = interval, Kelembaban = rasio).")
st.markdown(f'<div class="footer-id">{NAMA} &nbsp;|&nbsp; NIM {NIM} &nbsp;|&nbsp; Bedah Kasus 3: Dataset Pengamatan Cuaca</div>',
            unsafe_allow_html=True)

if f.empty:
    st.warning("Tidak ada pengamatan yang cocok dengan filter. Longgarkan pilihan kota, kondisi, atau slider suhu/kelembaban di sidebar.")
    st.stop()

# ----------------------------------------------------------------------------- KPI
k = st.columns(5)
k[0].metric("Jumlah pengamatan", id_num(len(f)), f"{len(f) / len(df):.0%} dari seluruh data", delta_color="off")
k[1].metric(f"Rata-rata suhu ({SATUAN})", id_num(f[TS].mean(), 1), f"median {id_num(f[TS].median(), 1)}", delta_color="off")
k[2].metric("Rata-rata kelembaban", f"{id_num(f[RH].mean(), 1)}%", f"median {f[RH].median():.0f}%", delta_color="off")
k[3].metric("Pengamatan hujan", f"{f['Hujan'].mean():.1%}", "Hujan + Hujan Petir", delta_color="off")
rho_all = f[SUHU].corr(f[RH]) if len(f) > 2 else float("nan")
k[4].metric("Korelasi suhu-kelembaban", "-" if np.isnan(rho_all) else f"{rho_all:+.2f}", "Pearson r", delta_color="off")

tabs = st.tabs(["Ringkasan", "Kota & Kondisi", "Suhu", "Kelembaban", "Hubungan", "Data"])

# ============================================================================= TAB 1 RINGKASAN
with tabs[0]:
    c1, c2, c3 = st.columns([1, 1, 1.15])
    kc = f["Kota"].value_counts().reset_index()
    kc.columns = ["Kota", "Jumlah"]
    fig = px.bar(kc.sort_values("Jumlah"), x="Jumlah", y="Kota", orientation="h", color_discrete_sequence=[BIRU], text="Jumlah")
    fig.update_traces(textposition="outside", cliponaxis=False)
    c1.plotly_chart(rapikan(fig, 400, "Pengamatan per Kota"), width="stretch", key="r_kota")

    ko = f["Kondisi"].value_counts().reset_index()
    ko.columns = ["Kondisi", "Jumlah"]
    fig = px.pie(ko, names="Kondisi", values="Jumlah", hole=.5, color="Kondisi", color_discrete_map=WARNA_KONDISI,
                 category_orders={"Kondisi": KONDISI})
    fig.update_traces(textinfo="percent", sort=False)
    c2.plotly_chart(rapikan(fig, 400, "Komposisi Kondisi"), width="stretch", key="r_kon")

    ag = f.groupby("Kota").agg(T=(TS, "mean"), H=(RH, "mean"), n=(RH, "size")).reset_index()
    fig = px.scatter(ag, x="T", y="H", text="Kota", size="n", color_discrete_sequence=[ORANYE], labels={"T": f"Suhu rata-rata ({SATUAN})", "H": "Kelembaban rata-rata (%)"})
    fig.update_traces(textposition="top center", marker=dict(color=ORANYE, line=dict(color="#6b3a05", width=1)))
    c3.plotly_chart(rapikan(fig, 400, "Pusat iklim tiap kota"), width="stretch", key="r_pusat")

    t_k = f.groupby("Kota")[TS].mean()
    h_k = f.groupby("Kota")[RH].mean()
    st.info(f"**Temuan cepat.** Kota terpanas: **{t_k.idxmax()}** ({id_num(t_k.max(), 1)} {SATUAN}); tersejuk: **{t_k.idxmin()}** ({id_num(t_k.min(), 1)} {SATUAN}). "
            f"Kota paling lembap: **{h_k.idxmax()}** ({id_num(h_k.max(), 1)}%). Kondisi paling sering: **{f['Kondisi'].mode().iloc[0]}**.")

# ============================================================================= TAB 2 KOTA & KONDISI (NOMINAL)
with tabs[1]:
    tag("<b>Kategorik (Nominal)</b>: Kota dan Kondisi berupa label teks tanpa urutan. Visualisasi yang sesuai: Bar chart atau Pie chart.", BIRU)
    a, b, c = st.columns(3)
    var = a.radio("Variabel", ["Kota", "Kondisi"], horizontal=True)
    jenis = b.radio("Jenis grafik", ["Bar chart", "Pie chart"], horizontal=True)
    opsi_ukuran = ["Jumlah pengamatan", f"Rata-rata suhu ({SATUAN})", "Rata-rata kelembaban (%)", "% pengamatan hujan"]
    if jenis == "Bar chart":
        ukuran = c.selectbox("Ukuran", opsi_ukuran)
    else:
        ukuran = "Jumlah pengamatan"
        c.caption("Pie chart hanya untuk komposisi (jumlah pengamatan).")

    agg = f.groupby(var).agg(**{"Jumlah pengamatan": (RH, "size"), opsi_ukuran[1]: (TS, "mean"), opsi_ukuran[2]: (RH, "mean"),
                                opsi_ukuran[3]: ("Hujan", lambda s: s.mean() * 100)})
    agg["Persen"] = agg["Jumlah pengamatan"] / agg["Jumlah pengamatan"].sum() * 100
    if var == "Kondisi":
        agg = agg.reindex([x for x in KONDISI if x in agg.index] + [x for x in agg.index if x not in KONDISI])
    d = (agg.sort_values(ukuran) if var == "Kota" else agg).reset_index()

    c1, c2 = st.columns([1.1, 1])
    if jenis == "Bar chart":
        if var == "Kondisi":
            fig = px.bar(d, x="Kondisi", y=ukuran, color="Kondisi", color_discrete_map=WARNA_KONDISI, text=ukuran)
            fig.update_layout(showlegend=False)
        else:
            fig = px.bar(d, x=ukuran, y="Kota", orientation="h", color_discrete_sequence=[BIRU], text=ukuran)
        fig.update_traces(texttemplate="%{text:,.0f}" if ukuran == "Jumlah pengamatan" else "%{text:.1f}", textposition="outside", cliponaxis=False)
    else:
        if var == "Kondisi":
            fig = px.pie(d, names="Kondisi", values=ukuran, hole=.45, color="Kondisi", color_discrete_map=WARNA_KONDISI)
        else:
            fig = px.pie(d.sort_values(ukuran, ascending=False), names="Kota", values=ukuran, hole=.45, color_discrete_sequence=BLUES)
        fig.update_traces(textinfo="percent+label", sort=False)
    c1.plotly_chart(rapikan(fig, 440, f"{ukuran} per {var}"), width="stretch", key="n_main")

    ct = pd.crosstab(f["Kota"], f["Kondisi"]).reindex(columns=[x for x in KONDISI if x in set(f["Kondisi"])] + sorted(set(f["Kondisi"]) - set(KONDISI)))
    mode = c2.radio("Heatmap Kota x Kondisi", ["% per kota", "Jumlah"], horizontal=True)
    hm = ct.div(ct.sum(axis=1), axis=0) * 100 if mode == "% per kota" else ct
    fig = px.imshow(hm, text_auto=".0f", color_continuous_scale="Blues", aspect="auto", labels=dict(x="", y="", color="%" if mode != "Jumlah" else "n"))
    c2.plotly_chart(rapikan(fig, 380, "Kota x Kondisi"), width="stretch", key="n_heat")

    st.markdown(f"**Modus {var}:** {agg['Jumlah pengamatan'].idxmax()} ({int(agg['Jumlah pengamatan'].max())} pengamatan). Untuk data nominal, hanya frekuensi dan modus yang bermakna sebagai ukuran pemusatan.")
    st.dataframe(agg.reset_index(), hide_index=True, width="stretch", column_config={
        opsi_ukuran[1]: st.column_config.NumberColumn(format="%.1f"), opsi_ukuran[2]: st.column_config.NumberColumn(format="%.1f"),
        opsi_ukuran[3]: st.column_config.NumberColumn(format="%.1f%%"),
        "Persen": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)})

# ============================================================================= TAB 3 SUHU (INTERVAL)
with tabs[2]:
    tag("<b>Numerik - Interval</b>: Suhu. Jarak antar suhu sama, tetapi 0 °C bukan 'tidak ada suhu', jadi aturan rasio tidak berlaku. Visualisasi: Histogram atau Box plot.", ORANYE)
    a, b, c = st.columns(3)
    jenis = a.radio("Jenis grafik", ["Histogram", "Box plot", "Violin"], horizontal=True, key="t_jenis")
    grup = b.selectbox("Kelompokkan menurut", ["Tidak ada", "Kota", "Kondisi"], key="t_grup")
    bins = c.slider("Jumlah bin (histogram)", 10, 60, 30, key="t_bins", disabled=jenis != "Histogram")
    g = None if grup == "Tidak ada" else grup

    c1, c2 = st.columns([1.3, 1])
    if jenis == "Histogram":
        fig = px.histogram(f, x=TS, nbins=bins, marginal="box", barmode="stack", **warna_grup(g, ORANYE))
        fig.add_vline(x=f[TS].mean(), line_dash="dash", line_color="#C0392B")
        fig.add_vline(x=f[TS].median(), line_dash="dash", line_color="#27AE60")
        c1.caption("Garis putus-putus merah = mean, hijau = median.")
    else:
        fn = px.box if jenis == "Box plot" else px.violin
        kw = dict(box=True, points=False) if jenis == "Violin" else dict(points="outliers")
        if g:
            fig = fn(f, x=g, y=TS, **warna_grup(g, ORANYE), **kw)
            fig.update_layout(boxmode="overlay", violinmode="overlay", showlegend=False)
            if g == "Kota":
                fig.update_xaxes(categoryorder="median ascending")
        else:
            fig = fn(f, y=TS, color_discrete_sequence=[ORANYE], **kw)
    c1.plotly_chart(rapikan(fig, 470, f"Suhu: {jenis.lower()}" + (f" menurut {grup}" if g else "")), width="stretch", key="t_main")

    fig = qq_fig(f[TS], ORANYE)
    fig.update_yaxes(title=TS)
    c2.plotly_chart(rapikan(fig, 470, "QQ-plot: seberapa normal suhu?"), width="stretch", key="t_qq")

    s = f[TS]
    q1, q3 = s.quantile([.25, .75])
    m = st.columns(6)
    m[0].metric("Mean", id_num(s.mean(), 2))
    m[1].metric("Median", id_num(s.median(), 2))
    m[2].metric("Simpangan baku", id_num(s.std(), 2) if len(s) > 1 else "-")
    m[3].metric("Skewness", f"{s.skew():.2f}" if len(s) > 2 else "-")
    m[4].metric("IQR", id_num(q3 - q1, 2))
    m[5].metric("Koefisien variasi", "tidak sah", help="CV (SD/mean) hanya bermakna pada skala rasio. Suhu bersifat interval.")

    with st.expander("Uji sendiri: mengapa rasio suhu tidak bermakna?", expanded=False):
        e1, e2 = st.columns(2)
        va = e1.number_input("Suhu A (°C)", value=10.0, step=1.0)
        vb = e2.number_input("Suhu B (°C)", value=20.0, step=1.0)
        tab = pd.DataFrame({"Satuan": ["°C", "K (nol absolut)", "°F"],
                            "Suhu A": [va, va + 273.15, va * 9 / 5 + 32], "Suhu B": [vb, vb + 273.15, vb * 9 / 5 + 32]})
        tab["Rasio B / A"] = np.where(tab["Suhu A"] != 0, tab["Suhu B"] / tab["Suhu A"].replace(0, np.nan), np.nan)
        st.dataframe(tab, hide_index=True, width="stretch", column_config={
            "Suhu A": st.column_config.NumberColumn(format="%.2f"), "Suhu B": st.column_config.NumberColumn(format="%.2f"),
            "Rasio B / A": st.column_config.NumberColumn(format="%.3f")})
        st.caption("Rasio berubah menurut satuan, sehingga 'B dua kali lebih panas dari A' tidak bermakna. "
                   "Selisih (B - A) tetap bermakna: dalam K sama dengan °C, dan dalam °F selalu 1,8 kali selisih °C.")

# ============================================================================= TAB 4 KELEMBABAN (RASIO)
with tabs[3]:
    tag("<b>Numerik - Rasio</b>: Kelembaban. 0% berarti benar-benar tidak ada kelembaban (nol absolut), sehingga rasio dan koefisien variasi bermakna. Visualisasi: Histogram atau Box plot.", ORANYE_TUA)
    a, b, c = st.columns(3)
    jenis = a.radio("Jenis grafik", ["Histogram", "Box plot", "Violin"], horizontal=True, key="h_jenis")
    grup = b.selectbox("Kelompokkan menurut", ["Tidak ada", "Kota", "Kondisi"], key="h_grup")
    bins = c.slider("Jumlah bin (histogram)", 10, 60, 25, key="h_bins", disabled=jenis != "Histogram")
    g = None if grup == "Tidak ada" else grup
    ambang = st.slider("Ambang kelembapan (%)", 40, 100, 80, help="Dipakai untuk menghitung proporsi pengamatan lembap dan garis pada ECDF.")

    c1, c2 = st.columns([1.3, 1])
    if jenis == "Histogram":
        fig = px.histogram(f, x=RH, nbins=bins, marginal="box", barmode="stack", **warna_grup(g, ORANYE_TUA))
        fig.add_vline(x=f[RH].mean(), line_dash="dash", line_color="#C0392B")
        fig.add_vline(x=f[RH].median(), line_dash="dash", line_color="#27AE60")
        fig.add_vline(x=ambang, line_dash="dot", line_color="#555")
        c1.caption("Merah = mean, hijau = median, abu-abu = ambang kelembapan.")
    else:
        fn = px.box if jenis == "Box plot" else px.violin
        kw = dict(box=True, points=False) if jenis == "Violin" else dict(points="outliers")
        if g:
            fig = fn(f, x=g, y=RH, **warna_grup(g, ORANYE_TUA), **kw)
            fig.update_layout(boxmode="overlay", violinmode="overlay", showlegend=False)
            if g == "Kota":
                fig.update_xaxes(categoryorder="median ascending")
        else:
            fig = fn(f, y=RH, color_discrete_sequence=[ORANYE_TUA], **kw)
        fig.add_hline(y=ambang, line_dash="dot", line_color="#555")
    c1.plotly_chart(rapikan(fig, 470, f"Kelembaban: {jenis.lower()}" + (f" menurut {grup}" if g else "")), width="stretch", key="h_main")

    xs = np.sort(f[RH].to_numpy())
    fig = go.Figure(go.Scatter(x=xs, y=np.arange(1, len(xs) + 1) / len(xs), mode="lines", line=dict(color=ORANYE_TUA, width=3, shape="hv"), name="ECDF"))
    p_amb = (f[RH] <= ambang).mean()
    fig.add_vline(x=ambang, line_dash="dot", line_color="#555", annotation_text=f"RH <= {ambang}%: {p_amb:.0%}")
    fig.update_xaxes(title="Kelembaban (%)")
    fig.update_yaxes(title="Proporsi kumulatif")
    c2.plotly_chart(rapikan(fig, 470, "ECDF Kelembaban"), width="stretch", key="h_ecdf")

    s = f[RH]
    m = st.columns(6)
    m[0].metric("Mean", f"{id_num(s.mean(), 1)}%")
    m[1].metric("Median", f"{s.median():.0f}%")
    m[2].metric("Simpangan baku", id_num(s.std(), 1) if len(s) > 1 else "-")
    m[3].metric("Skewness", f"{s.skew():.2f}" if len(s) > 2 else "-", help="< 0: miring ke kiri karena batas atas 100%.")
    m[4].metric("Koefisien variasi", f"{s.std() / s.mean() * 100:.1f}%" if len(s) > 1 else "-", help="Sah karena kelembaban berskala rasio.")
    m[5].metric(f"RH >= {ambang}%", f"{(s >= ambang).mean():.1%}")

    lem = (f.assign(L=f[RH] >= ambang).groupby("Kota")["L"].mean() * 100).sort_values().reset_index()
    fig = px.bar(lem, x="L", y="Kota", orientation="h", color_discrete_sequence=[ORANYE_TUA], text="L", labels={"L": f"% pengamatan RH >= {ambang}%"})
    fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside", cliponaxis=False)
    st.plotly_chart(rapikan(fig, 340, f"Persentase pengamatan lembap (RH >= {ambang}%) per Kota"), width="stretch", key="h_lem")

# ============================================================================= TAB 5 HUBUNGAN
with tabs[4]:
    tag("<b>Hubungan antar variabel</b>: Suhu (interval) dan Kelembaban (rasio). Korelasi Pearson dan Spearman dapat dihitung untuk keduanya.", ORANYE)
    a, b, c = st.columns(3)
    tampil = a.radio("Tampilan", ["Scatter", "Kepadatan"], horizontal=True)
    warna = b.selectbox("Warna titik menurut", ["Kondisi", "Kota", "Tidak ada"], disabled=tampil != "Scatter")
    garis = c.toggle("Garis regresi linear", value=True)

    if tampil == "Scatter":
        arg = warna_grup(None if warna == "Tidak ada" else warna, ORANYE)
        fig = px.scatter(f, x=TS, y=RH, opacity=.7, hover_data=["Kota", "Kondisi"], **arg)
    else:
        fig = px.density_heatmap(f, x=TS, y=RH, nbinsx=25, nbinsy=20, color_continuous_scale="Oranges")
    if garis and len(f) > 2 and f[TS].nunique() > 1:
        kof = np.polyfit(f[TS], f[RH], 1)
        x_lo = f[TS].min() if kof[0] >= 0 else max(f[TS].min(), (100 - kof[1]) / kof[0])    # garis dipotong di RH = 100%
        xs = np.array([x_lo, f[TS].max()])
        fig.add_trace(go.Scatter(x=xs, y=kof[0] * xs + kof[1], mode="lines", line=dict(color="#C0392B", width=3), name="Regresi linear"))
    c1, c2 = st.columns([1.3, 1])
    c1.plotly_chart(rapikan(fig, 470, "Suhu vs Kelembaban"), width="stretch", key="x_sc")

    if len(f) > 2:
        pear, spear = f[SUHU].corr(f[RH]), f[SUHU].corr(f[RH], method="spearman")
        rk = {}
        for kt, gg in f.groupby("Kota"):
            if len(gg) > 5 and gg[SUHU].nunique() > 1:
                rk[kt] = gg[SUHU].corr(gg[RH])
        rk = pd.Series(rk).sort_values().reset_index()
        rk.columns = ["Kota", "r"]
        if not rk.empty:
            fig = px.bar(rk, x="r", y="Kota", orientation="h", color_discrete_sequence=[ORANYE], text="r")
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
            c2.plotly_chart(rapikan(fig, 470, "Korelasi di dalam tiap kota (Pearson r)"), width="stretch", key="x_rkota")
        m = st.columns(3)
        m[0].metric("Pearson r", f"{pear:+.2f}", help="Korelasi linear. Tidak berubah oleh pergantian satuan suhu.")
        m[1].metric("Spearman rho", f"{spear:+.2f}", help="Korelasi peringkat.")
        if garis and f[TS].nunique() > 1:
            m[2].metric(f"Kemiringan (% per 1 {SATUAN})", f"{kof[0]:+.2f}")
    else:
        c2.info("Data terlalu sedikit untuk menghitung korelasi.")

    st.markdown("**Rata-rata per kelompok (95% CI)**")
    d1, d2 = st.columns(2)
    gk = d1.radio("Kelompok", ["Kondisi", "Kota"], horizontal=True, key="x_gk")
    vk = d2.radio("Variabel", [f"Suhu ({SATUAN})", "Kelembaban (%)"], horizontal=True, key="x_vk")
    kolom = TS if vk.startswith("Suhu") else RH
    gs = f.groupby(gk)[kolom].agg(["mean", "std", "count"]).reset_index()
    gs["ci"] = 1.96 * gs["std"].fillna(0) / np.sqrt(gs["count"])
    gs = gs.set_index(gk).reindex([x for x in KONDISI if x in set(gs[gk])]).reset_index() if gk == "Kondisi" else gs.sort_values("mean")
    if gk == "Kondisi":
        fig = px.bar(gs, x=gk, y="mean", error_y="ci", color=gk, color_discrete_map=WARNA_KONDISI, labels={"mean": vk})
        fig.update_layout(showlegend=False)
    else:
        fig = px.bar(gs, x="mean", y=gk, error_x="ci", orientation="h", color_discrete_sequence=[ORANYE], labels={"mean": vk})
    st.plotly_chart(rapikan(fig, 380, f"Rata-rata {vk} per {gk}"), width="stretch", key="x_grup")
    st.caption("Sumbu tidak dipotong. Garis error = 95% interval kepercayaan.")

# ============================================================================= TAB 6 DATA
with tabs[5]:
    st.markdown(f"**{id_num(len(f))} pengamatan** sesuai filter dari {id_num(len(df))} total.")
    st.dataframe(f[["Kota", SUHU, RH, "Kondisi"]], hide_index=True, width="stretch", column_config={
        SUHU: st.column_config.NumberColumn(format="%.1f"), RH: st.column_config.NumberColumn(format="%d")})
    st.download_button("Unduh data terfilter (CSV)", f[KOLOM_WAJIB].to_csv(index=False).encode("utf-8"),
                       file_name="data_cuaca_terfilter.csv", mime="text/csv")
    st.markdown("**Kamus data dan pilihan visualisasi**")
    st.dataframe(pd.DataFrame({
        "Kolom": ["Kota", SUHU, RH, "Kondisi"],
        "Skala": ["Kategorik - Nominal", "Numerik - Interval", "Numerik - Rasio", "Kategorik - Nominal"],
        "Contoh": ["Jakarta", "32,5", "75", "Cerah"],
        "Catatan": ["Label teks tanpa urutan", "0 °C bukan nol absolut; rasio dan CV tidak sah", "0% = nol absolut; rasio dan CV sah", "Label teks tanpa urutan"],
        "Visualisasi": ["Bar / Pie", "Histogram / Box plot", "Histogram / Box plot", "Bar / Pie"]}),
        hide_index=True, width="stretch")
