"""
PBH Explorer — an interactive companion to the literature review
"Primordial Black Holes in the Early Universe: Investigating Formation
Mechanisms and Potential Chemical Signatures" (Armeen Zahra).

Run locally:   streamlit run app.py
Deploy:        push this folder to GitHub, then deploy on share.streamlit.io

All curves are simplified, order-of-magnitude teaching models. They are
built to show the *trends* discussed in the review, not to replace the
published constraint plots.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.special import erfc

# ---------------------------------------------------------------------------
# Page setup + theme (blue galaxy palette, Comic Sans everywhere)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PBH Explorer",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palette: only blues, cyans, indigos, violets, teals and white.
C = {
    "bg": "#050b1f",
    "panel": "#0b1636",
    "ice": "#e6f0ff",
    "cyan": "#4cc9f0",
    "sky": "#4895ef",
    "royal": "#4361ee",
    "indigo": "#3f37c9",
    "violet": "#9d8df1",
    "lavender": "#b8c0ff",
    "teal": "#56cfe1",
    "aqua": "#80ffdb",
    "mist": "#a9def9",
    "deep": "#1b2a6b",
}
FONT = "'Comic Sans MS', 'Comic Sans', 'Comic Neue', cursive"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Comic+Neue:wght@400;700&display=swap');

html, body, .stApp, .stMarkdown, p, li, label, input, textarea, button, select,
h1, h2, h3, h4, h5, h6, td, th, div, span, [class*="st-"] {{
    font-family: {FONT} !important;
}}
/* keep Streamlit's icon font working (expander arrows, etc.) */
[data-testid="stIconMaterial"], .material-symbols-rounded,
span[class*="material-symbols"] {{
    font-family: 'Material Symbols Rounded' !important;
}}

.stApp {{
    background:
      radial-gradient(1px 1px at 12% 18%, #ffffffcc 50%, transparent 51%),
      radial-gradient(1px 1px at 72% 12%, #ffffffaa 50%, transparent 51%),
      radial-gradient(1.5px 1.5px at 33% 72%, #cfe8ffcc 50%, transparent 51%),
      radial-gradient(1px 1px at 88% 64%, #ffffffaa 50%, transparent 51%),
      radial-gradient(1px 1px at 54% 40%, #e6f0ff99 50%, transparent 51%),
      radial-gradient(1.5px 1.5px at 6% 88%, #ffffffbb 50%, transparent 51%),
      radial-gradient(1px 1px at 93% 30%, #b8c0ffcc 50%, transparent 51%),
      radial-gradient(1px 1px at 41% 93%, #ffffff99 50%, transparent 51%),
      radial-gradient(ellipse 60% 45% at 85% 5%, rgba(67,97,238,0.28), transparent 70%),
      radial-gradient(ellipse 55% 40% at 5% 95%, rgba(157,141,241,0.20), transparent 70%),
      radial-gradient(ellipse 70% 50% at 50% 50%, rgba(76,201,240,0.06), transparent 70%),
      {C['bg']};
    background-attachment: fixed;
}}
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0b1636 0%, #070f28 100%);
    border-right: 1px solid rgba(76,201,240,0.25);
}}
h1, h2, h3 {{ color: {C['mist']} !important; }}
.hero {{
    padding: 1.4rem 1.6rem; border-radius: 22px;
    background: linear-gradient(120deg, rgba(27,42,107,0.85), rgba(63,55,201,0.55) 55%, rgba(76,201,240,0.25));
    border: 1px solid rgba(128,255,219,0.25);
    box-shadow: 0 0 40px rgba(67,97,238,0.25);
    margin-bottom: 1rem;
}}
.hero h1 {{ margin: 0 0 .3rem 0; font-size: 2.3rem; color: #ffffff !important; }}
.hero p  {{ margin: 0; color: {C['mist']}; font-size: 1.05rem; }}
.card {{
    background: rgba(11,22,54,0.78); border: 1px solid rgba(76,201,240,0.28);
    border-radius: 16px; padding: 0.9rem 1.1rem; margin: 0.35rem 0;
}}
.card .lbl {{ color: {C['lavender']}; font-size: .85rem; }}
.card .val {{ color: #ffffff; font-size: 1.35rem; font-weight: 700; }}
.card .sub {{ color: {C['teal']}; font-size: .82rem; }}
.note {{
    background: rgba(67,97,238,0.14); border-left: 4px solid {C['cyan']};
    border-radius: 10px; padding: .75rem 1rem; margin: .6rem 0; color: {C['ice']};
}}
.good {{ border-left-color: {C['aqua']}; background: rgba(128,255,219,0.10); }}
.dim  {{ border-left-color: {C['violet']}; background: rgba(157,141,241,0.12); }}
.paper {{
    background: rgba(157,141,241,0.10); border: 1px dashed rgba(184,192,255,0.5);
    border-radius: 12px; padding: .6rem .9rem; color: {C['lavender']}; font-size: .92rem;
}}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; flex-wrap: wrap; }}
.stTabs [data-baseweb="tab"] {{
    background: rgba(11,22,54,0.75); border-radius: 12px 12px 0 0;
    padding: .45rem .9rem; color: {C['mist']};
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(180deg, rgba(67,97,238,0.6), rgba(11,22,54,0.9));
    color: #ffffff !important;
}}
table.cmp {{ width: 100%; border-collapse: collapse; }}
table.cmp th {{ background: rgba(67,97,238,0.35); color: #fff; padding: .5rem; text-align: left; }}
table.cmp td {{ border-bottom: 1px solid rgba(76,201,240,0.2); padding: .5rem; color: {C['ice']}; vertical-align: top; }}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Physical constants (cgs unless stated)
# ---------------------------------------------------------------------------
G_CGS = 6.674e-8
C_CGS = 2.998e10
HBAR = 1.055e-27
M_SUN_G = 1.989e33
GAMMA = 0.2                       # fraction of horizon mass that collapses
C3_OVER_G = C_CGS**3 / G_CGS      # g/s
RHO_DM_LOCAL = 7.1e-25            # g/cm^3  (≈0.4 GeV/cm^3)
AU_CM = 1.496e13
YEAR_S = 3.156e7
WINDOW_REVIEW = (1e17, 1e25)      # window quoted in the review
WINDOW_RECENT = (1e17, 1e22)      # approximate window in recent compilations

REFERENCE_OBJECTS = [
    (1.6e17, "Mount Everest"),
    (1e20, "a large asteroid (~1 km³ of rock)"),
    (9.4e23, "the dwarf planet Ceres"),
    (7.35e25, "the Moon"),
    (5.97e27, "the Earth"),
    (1.9e30, "Jupiter"),
    (M_SUN_G, "the Sun"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sci(x, digits=2):
    """Pretty scientific notation using superscripts."""
    if x == 0 or not np.isfinite(x):
        return str(x)
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / 10**exp
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    if abs(mant - 1) < 0.005:
        return f"10{str(exp).translate(sup)}"
    return f"{mant:.{digits-1}f}×10{str(exp).translate(sup)}"


def card(label, value, sub=""):
    st.markdown(
        f'<div class="card"><div class="lbl">{label}</div>'
        f'<div class="val">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def note(text, kind=""):
    st.markdown(f'<div class="note {kind}">{text}</div>', unsafe_allow_html=True)


def paper(text):
    st.markdown(f'<div class="paper">📘 <b>In the review:</b> {text}</div>', unsafe_allow_html=True)


def style_fig(fig, height=460, xlog=False, ylog=False, xtitle="", ytitle=""):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,16,44,0.65)",
        font=dict(family=FONT, color=C["ice"], size=14),
        margin=dict(l=60, r=20, t=40, b=55),
        legend=dict(bgcolor="rgba(11,22,54,0.7)", bordercolor="rgba(76,201,240,0.3)", borderwidth=1),
        hoverlabel=dict(font_family=FONT, bgcolor=C["deep"]),
        colorway=[C["cyan"], C["violet"], C["aqua"], C["sky"], C["lavender"], C["teal"]],
    )
    axis = dict(gridcolor="rgba(169,222,249,0.12)", zerolinecolor="rgba(169,222,249,0.2)",
                linecolor="rgba(169,222,249,0.3)", exponentformat="power")
    fig.update_xaxes(**axis, title=xtitle, type="log" if xlog else "linear")
    fig.update_yaxes(**axis, title=ytitle, type="log" if ylog else "linear")
    return fig


def compare_object(m_g):
    best = min(REFERENCE_OBJECTS, key=lambda o: abs(np.log10(o[0]) - np.log10(m_g)))
    ratio = m_g / best[0]
    if 0.5 < ratio < 2:
        return f"about the mass of {best[1]}"
    return f"{sci(ratio)} × the mass of {best[1]}"


# --- basic PBH properties --------------------------------------------------
def schwarzschild_cm(m_g):
    return 2 * G_CGS * m_g / C_CGS**2


def hawking_T_GeV(m_g):
    return 1.06e13 / m_g


def lifetime_yr(m_g):
    return 13.8e9 * (m_g / 5.1e14) ** 3


def formation_time_s(m_g):
    return m_g / (GAMMA * C3_OVER_G)


def formation_T_MeV(t_s):
    gstar = 106.75
    T = np.sqrt(2.42 / (np.sqrt(gstar) * t_s))
    if T < 1e5:  # below ~100 GeV, fewer relativistic species
        gstar = 10.75
        T = np.sqrt(2.42 / (np.sqrt(gstar) * t_s))
    return T


def gstar_of_mass(m_g):
    return np.where(m_g < 1e30, 106.75, 10.75)


# --- simplified observational constraints on f_PBH (log-parabola fits) -----
@st.cache_data
def constraints():
    x = np.linspace(9, 40, 1200)  # log10 M [g]
    M = 10**x
    big = 50.0
    curves = {}

    evap = np.full_like(x, big)
    sel = x > 14.7
    evap[sel] = -8 + 3.3 * (x[sel] - 15)
    evap[x <= 14.7] = -9
    curves["Hawking evaporation (γ-rays, CMB)"] = (evap, C["violet"])

    hsc = np.where((x > 21.3) & (x < 28), -2.6 + 1.1 * (x - 23.6) ** 2, big)
    curves["Subaru HSC microlensing (M31)"] = (hsc, C["cyan"])

    ogle = np.where((x > 26.5) & (x < 30.5), -2.0 + 0.55 * (x - 28.5) ** 2, big)
    curves["OGLE microlensing (bulge)"] = (ogle, C["teal"])

    eros = np.where((x > 26) & (x < 35.5), -1.25 + 0.16 * (x - 31) ** 2, big)
    curves["EROS / MACHO microlensing"] = (eros, C["sky"])

    ligo = np.where((x > 32.5) & (x < 36.5), -3.0 + 0.6 * (x - 34.5) ** 2, big)
    curves["LIGO/Virgo merger rates"] = (ligo, C["lavender"])

    cmb = np.where(x > 33, -1.0 - 1.4 * (x - 34), big)
    cmb = np.maximum(cmb, -9)
    curves["CMB accretion (Planck)"] = (np.where(x > 33, cmb, big), C["royal"])

    dyn = np.where(x > 36, -1.0 - 0.3 * (x - 36), big)
    curves["Dynamical (dwarf galaxies, binaries)"] = (dyn, C["mist"])

    combined = np.min(np.vstack([v[0] for v in curves.values()]), axis=0)
    return x, M, curves, combined


def f_max_at(m_g):
    x, _, _, comb = constraints()
    return 10 ** np.interp(np.log10(m_g), x, comb)


def constraint_traces(fig, show_legend=True, opacity=0.22):
    x, M, curves, combined = constraints()
    top = 10.0
    for name, (logf, col) in curves.items():
        y = 10 ** np.clip(logf, -12, 1)
        mask = logf < 1
        if not mask.any():
            continue
        xs, ys = M[mask], y[mask]
        fig.add_trace(go.Scatter(
            x=np.concatenate([xs, xs[::-1]]), y=np.concatenate([ys, np.full_like(ys, top)]),
            fill="toself", fillcolor=hex_alpha(col, opacity), line=dict(color=col, width=1.6),
            name=name, hoverinfo="name", showlegend=show_legend,
        ))
    return fig


def hex_alpha(hx, a):
    hx = hx.lstrip("#")
    r, g, b = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


# --- inflation: lognormal peak in the curvature power spectrum -------------
def mass_to_k(m_g, gstar=106.75):
    """Horizon-entry wavenumber (Mpc^-1) for a PBH of mass m_g."""
    m_sun = m_g / M_SUN_G
    return 1e6 * np.sqrt(GAMMA * 17 * (gstar / 10.75) ** (-1 / 6) / m_sun)


def k_to_hz(k):
    return 1.55e-15 * k


def beta_radiation(sigma, delta_c):
    return 0.5 * erfc(delta_c / (np.sqrt(2) * np.maximum(sigma, 1e-30)))


def beta_dust(sigma):
    # Khlopov–Polnarev-type estimate for pressureless (dust-like) collapse
    return 0.056 * sigma**5


def beta_to_f(beta, m_g):
    gs = gstar_of_mass(m_g)
    return 1.68e8 * (gs / 10.75) ** (-0.25) * (m_g / M_SUN_G) ** (-0.5) * beta


def inflation_spectrum(logM, A, width, logMp, delta_c):
    M = 10**logM
    k = mass_to_k(M)
    kp = mass_to_k(10**logMp)
    P = 2.1e-9 + A * np.exp(-np.log(k / kp) ** 2 / (2 * width**2))
    sigma = np.sqrt(16 / 81 * P)
    beta = beta_radiation(sigma, delta_c)
    f = beta_to_f(beta, M)
    return k, P, sigma, beta, f


def total_f(logM, f):
    # f is per ln M; integrate over ln M
    return np.trapezoid(f, logM * np.log(10)) if hasattr(np, "trapezoid") else np.trapz(f, logM * np.log(10))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌌 PBH Explorer")
    st.markdown(
        "An interactive companion to the review **“Primordial Black Holes in the Early "
        "Universe: Investigating Formation Mechanisms and Potential Chemical Signatures.”**"
    )
    st.markdown("**Author:** Armeen Zahra  \n**Mentor:** Ashley Walker")
    st.markdown("---")
    st.markdown("### 🧭 How to use")
    st.markdown(
        "Each tab matches a section of the review. Move the sliders and watch the "
        "physics respond. The 📘 boxes show what the review says about each idea."
    )
    st.markdown("---")
    st.markdown("### 🔢 Handy numbers")
    st.markdown(
        f"- Sun: {sci(M_SUN_G)} g\n- Moon: {sci(7.35e25)} g\n"
        f"- Dark-matter density near the Sun: ≈0.4 GeV/cm³\n"
        f"- Ordinary matter: ~5% of the cosmic energy budget\n"
        f"- Dark matter: ~27%"
    )
    st.markdown("---")
    st.caption(
        "All models here are simplified teaching versions that show the trends in the "
        "literature. For exact limits, see the published constraint plots "
        "(e.g. Carr & Kühnel 2020; Green & Kavanagh 2021)."
    )

st.markdown(
    """
<div class="hero">
  <h1>🕳️ Primordial Black Hole Explorer</h1>
  <p>Black holes born in the first second after the Big Bang. Could they be the dark matter?
  Explore how they form, how they change early-universe chemistry, and how we might catch them.</p>
</div>
""",
    unsafe_allow_html=True,
)

tabs = st.tabs([
    "🕳️ Mass Window",
    "🌌 Inflation Model",
    "🔄 Big Bounce Model",
    "🧪 Astrochemistry",
    "🌊 Gravitational Waves",
    "🔭 Microlensing",
    "📘 Review Map & Quiz",
])

# ===========================================================================
# TAB 1 — Mass window
# ===========================================================================
with tabs[0]:
    st.markdown("## The sub-lunar mass window")
    paper(
        "Section 1 explains the “process of elimination”: Hawking evaporation rules out the lightest "
        "PBHs, microlensing surveys rule out the heaviest, and a window remains where PBHs could be "
        "100% of dark matter (Figure 1)."
    )

    c1, c2 = st.columns([1, 2.3])
    with c1:
        logM = st.slider("PBH mass  (log₁₀ grams)", 9.0, 38.0, 20.0, 0.1, key="m_window")
        M = 10**logM
        fdm = st.select_slider(
            "Fraction of dark matter in PBHs  (f_PBH)",
            options=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.3, 1.0], value=1.0,
            format_func=lambda v: f"{v:g}",
        )
        show_recent = st.toggle("Show window from recent compilations", value=True)

    fmax = f_max_at(M)
    allowed = fdm <= fmax
    with c2:
        fig = go.Figure()
        constraint_traces(fig)
        fig.add_vrect(x0=WINDOW_REVIEW[0], x1=WINDOW_REVIEW[1], fillcolor=hex_alpha(C["aqua"], 0.06),
                      line_width=0, annotation_text="window quoted in review",
                      annotation_position="bottom left", annotation_font_color=C["aqua"])
        if show_recent:
            fig.add_vrect(x0=WINDOW_RECENT[0], x1=WINDOW_RECENT[1], fillcolor=hex_alpha(C["aqua"], 0.10),
                          line=dict(color=C["aqua"], dash="dot", width=1),
                          annotation_text="window in recent compilations",
                          annotation_position="top left", annotation_font_color=C["aqua"])
        fig.add_hline(y=1, line=dict(color=C["ice"], dash="dash", width=1),
                      annotation_text="all dark matter", annotation_font_color=C["ice"])
        fig.add_trace(go.Scatter(
            x=[M], y=[fdm], mode="markers",
            marker=dict(size=18, color=C["aqua"] if allowed else C["violet"], symbol="star",
                        line=dict(color="white", width=1.5)),
            name="your PBH",
        ))
        style_fig(fig, 500, True, True, "PBH mass (grams)", "max fraction of dark matter, f_PBH")
        fig.update_yaxes(range=[-10, 1])
        fig.update_xaxes(range=[9, 38])
        fig.update_layout(legend=dict(font=dict(size=11), orientation="h", y=-0.22))
        st.plotly_chart(fig, width="stretch")

    if allowed:
        note(f"✅ <b>Allowed!</b> A population of {sci(M)} g PBHs making up {fdm:g} of dark matter "
             f"is not excluded by these constraints (limit here ≈ {sci(min(fmax, 1))}).", "good")
    else:
        note(f"🚫 <b>Ruled out.</b> At {sci(M)} g, observations allow at most f_PBH ≈ {sci(fmax)}. "
             f"Lower the fraction or slide the mass into the window.", "dim")

    st.markdown("### What would one of these black holes be like?")
    r_s = schwarzschild_cm(M)
    T = hawking_T_GeV(M)
    tau = lifetime_yr(M)
    t_form = formation_time_s(M)
    T_form = formation_T_MeV(t_form)

    def length_str(cm):
        if cm < 1e-12:
            return f"{sci(cm*1e13)} fm", "smaller than an atomic nucleus"
        if cm < 1e-9:
            return f"{sci(cm*1e13)} fm", "between a nucleus and an atom"
        if cm < 1e-6:
            return f"{sci(cm*1e8)} Å", "about the size of an atom"
        if cm < 1e-1:
            return f"{sci(cm*1e4)} µm", "microscopic"
        if cm < 1e2:
            return f"{sci(cm)} cm", "fits in your hand"
        if cm < 1e5:
            return f"{sci(cm/1e2)} m", "building-sized"
        if cm < 1e8:
            return f"{sci(cm/1e5)} km", "city-sized"
        return f"{sci(cm/1e5)} km", "planet-sized or bigger"

    def temp_str(gev):
        if gev > 1:
            return f"{sci(gev)} GeV"
        if gev > 1e-3:
            return f"{sci(gev*1e3)} MeV"
        if gev > 1e-6:
            return f"{sci(gev*1e6)} keV"
        return f"{sci(gev*1e9)} eV"

    a, b, c, d = st.columns(4)
    with a:
        card("Mass", f"{sci(M)} g", compare_object(M))
    with b:
        v, s = length_str(r_s)
        card("Event-horizon radius", v, s)
    with c:
        card("Hawking temperature", temp_str(T), f"≈ {sci(T*1.16e13)} K")
    with d:
        if tau < 13.8e9:
            card("Evaporation lifetime", f"{sci(tau)} yr", "💨 already evaporated today")
        else:
            card("Evaporation lifetime", f"{sci(tau)} yr", f"{sci(tau/13.8e9)} × age of the Universe")

    a, b, c, d = st.columns(4)
    with a:
        card("Born at", f"{sci(t_form)} s", "after the Big Bang")
    with b:
        card("Universe temperature then", f"{sci(T_form)} MeV", f"≈ {sci(T_form*1.16e10)} K")
    n_local = fdm * RHO_DM_LOCAL / M
    sep_cm = n_local ** (-1 / 3)
    with c:
        card("Typical distance between PBHs", f"{sci(sep_cm/AU_CM)} AU",
             "near the Sun (if they are the dark matter)")
    rate = n_local * np.pi * AU_CM**2 * 2.5e7 * YEAR_S
    with d:
        card("Passing within 1 AU of the Sun", f"{sci(rate)} / yr", "at ~250 km/s")

    with st.expander("☄️ Oort Cloud check (from the abstract)"):
        r_oort = 5e4 * AU_CM
        n_in_oort = n_local * 4 / 3 * np.pi * r_oort**3
        st.markdown(
            f"If PBHs of {sci(M)} g make up a fraction {fdm:g} of dark matter, there would be about "
            f"**{sci(n_in_oort)} of them inside the Oort Cloud** (radius ≈ 50,000 AU) at any moment. "
            "Heavy ones passing through could nudge comets onto new orbits, which is the idea behind "
            "the comet-based detection method mentioned in the abstract."
        )

    st.markdown("### 💨 Hawking evaporation: why the lightest PBHs are gone")
    e1, e2 = st.columns([1, 2.3])
    with e1:
        logM0 = st.slider("Starting mass  (log₁₀ g)", 9.0, 17.0, 14.5, 0.1, key="evap")
        M0 = 10**logM0
        tau0 = lifetime_yr(M0)
        note(f"Lifetime ≈ <b>{sci(tau0)} years</b>. "
             + ("It has <b>already evaporated</b>. 💥" if tau0 < 13.8e9 else "It <b>survives to today</b>. 🌑"))
        note("Smaller black holes are hotter and shrink faster: lifetime ∝ M³.", "dim")
    with e2:
        t = np.logspace(-6, np.log10(min(tau0 * 1.0, 1e14)), 400)
        t = t[t < tau0]
        Mt = M0 * np.clip(1 - t / tau0, 0, 1) ** (1 / 3)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=Mt, mode="lines", line=dict(color=C["cyan"], width=3),
                                 fill="tozeroy", fillcolor=hex_alpha(C["cyan"], 0.12), name="mass"))
        fig.add_vline(x=13.8e9, line=dict(color=C["aqua"], dash="dash"),
                      annotation_text="today", annotation_font_color=C["aqua"])
        style_fig(fig, 340, True, True, "time since Big Bang (years)", "mass (g)")
        st.plotly_chart(fig, width="stretch")

# ===========================================================================
# TAB 2 — Inflation
# ===========================================================================
with tabs[1]:
    st.markdown("## Model 1: rapid inflation with a “peak”")
    paper(
        "Section 2.1 (Franciolini et al., 2026) describes PBHs forming from “peaks” in the inflationary "
        "fluctuations. Section 2.3 adds that the inflationary route relies on <i>fine-tuned</i> peaks "
        "in the primordial power spectrum."
    )
    note("Inflation leaves tiny ripples in density. On the CMB scale they are about 1 part in 100,000 "
         "(amplitude ≈ 2×10⁻⁹). To make black holes, the ripples on small scales need a "
         "<b>peak ~10 million times bigger</b>. When that region re-enters the horizon, any patch "
         "denser than a threshold δc collapses.")

    c1, c2 = st.columns([1, 2.3])
    with c1:
        logA = st.slider("Peak height  log₁₀(A)", -3.0, -1.0, -1.78, 0.01, key="infA",
                         help="Amplitude of the peak in the curvature power spectrum")
        width = st.slider("Peak width  Δ (in ln k)", 0.2, 2.0, 0.6, 0.05, key="infW")
        logMp = st.slider("Peak PBH mass  (log₁₀ g)", 15.0, 36.0, 20.0, 0.1, key="infM")
        delta_c = st.slider("Collapse threshold  δc", 0.40, 0.60, 0.45, 0.01, key="infdc")

    logMgrid = np.linspace(9, 40, 1500)
    k, P, sigma, beta, fM = inflation_spectrum(logMgrid, 10**logA, width, logMp, delta_c)
    ftot = total_f(logMgrid, fM)

    with c2:
        sub1, sub2 = st.tabs(["PBH mass distribution", "Power spectrum"])
        with sub1:
            fig = go.Figure()
            constraint_traces(fig, show_legend=False, opacity=0.15)
            yy = np.clip(fM, 1e-14, None)
            fig.add_trace(go.Scatter(x=10**logMgrid, y=yy, mode="lines", name="f_PBH(M) from inflation",
                                     line=dict(color=C["aqua"], width=3.5),
                                     fill="tozeroy", fillcolor=hex_alpha(C["aqua"], 0.18)))
            style_fig(fig, 430, True, True, "PBH mass (g)", "f_PBH per ln M")
            fig.update_yaxes(range=[-10, 1.5])
            fig.update_xaxes(range=[12, 38])
            st.plotly_chart(fig, width="stretch")
        with sub2:
            kk = np.logspace(-4, 18, 800)
            kp = mass_to_k(10**logMp)
            Pk = 2.1e-9 * (kk / 0.05) ** (0.965 - 1) + 10**logA * np.exp(-np.log(kk / kp) ** 2 / (2 * width**2))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=kk, y=Pk, mode="lines", line=dict(color=C["cyan"], width=3),
                                     name="P_ζ(k)"))
            fig.add_vrect(x0=1e-4, x1=1, fillcolor=hex_alpha(C["violet"], 0.12), line_width=0,
                          annotation_text="measured by CMB", annotation_font_color=C["lavender"])
            style_fig(fig, 430, True, True, "wavenumber k (Mpc⁻¹)", "curvature power spectrum P_ζ")
            st.plotly_chart(fig, width="stretch")

    a, b, c = st.columns(3)
    with a:
        card("Total PBH dark-matter fraction", sci(ftot) if ftot > 1e-30 else "≈ 0",
             "too many! (overcloses)" if ftot > 1.5 else ("✨ could be all dark matter" if ftot > 0.3 else "a minor component"))
    with b:
        card("Fraction of the universe that collapsed", sci(beta.max()) if beta.max() > 1e-300 else "≈ 0",
             "β at the peak, at formation")
    with c:
        card("Typical density ripple σ", f"{sigma.max():.3f}", f"needs δc/σ ≈ {delta_c/sigma.max():.1f}")

    st.markdown("### 🎯 The fine-tuning problem")
    st.markdown("How sensitive is the PBH abundance to the height of the peak? Each point below re-runs "
                "the model with a different peak height (other settings unchanged).")
    As = np.logspace(-3, -1, 120)
    ft = np.array([total_f(logMgrid, inflation_spectrum(logMgrid, a_, width, logMp, delta_c)[4]) for a_ in As])
    good = (ft > 0.1) & (ft < 1)
    fig = go.Figure()
    fig.add_hrect(y0=0.1, y1=1, fillcolor=hex_alpha(C["aqua"], 0.15), line_width=0,
                  annotation_text="useful: 10–100% of dark matter", annotation_font_color=C["aqua"])
    fig.add_trace(go.Scatter(x=As, y=np.clip(ft, 1e-40, 1e20), mode="lines",
                             line=dict(color=C["violet"], width=3), name="total f_PBH"))
    fig.add_trace(go.Scatter(x=[10**logA], y=[max(ftot, 1e-40)], mode="markers",
                             marker=dict(size=16, color=C["aqua"], symbol="star"), name="your setting"))
    style_fig(fig, 360, True, True, "peak height A", "total f_PBH")
    fig.update_yaxes(range=[-30, 8])
    st.plotly_chart(fig, width="stretch")
    if good.any():
        lo, hi = As[good].min(), As[good].max()
        note(f"Only peak heights between <b>{lo:.4f}</b> and <b>{hi:.4f}</b> give a sensible amount of dark "
             f"matter, a change of just <b>{(hi/lo-1)*100:.0f}%</b>. Make the peak 2× smaller and PBHs "
             "basically vanish; make it 2× bigger and they overfill the universe. That is what "
             "“fine-tuned peaks” means in Section 2.3.")

# ===========================================================================
# TAB 3 — Bounce
# ===========================================================================
with tabs[2]:
    st.markdown("## Model 2: the bouncing universe")
    paper(
        "Section 2.2 (Ye et al., 2026) replaces the initial singularity with a <i>bounce</i>: a contracting "
        "universe reaches maximum density, then expands. PBHs form as gravity overwhelms pressure near "
        "the bounce point, and the scale of the bounce sets the PBH mass distribution (Figure 2)."
    )

    c1, c2 = st.columns([1, 2.3])
    with c1:
        w = st.slider("Equation of state while contracting  (w)", 0.0, 0.33, 0.0, 0.01, key="bw",
                      help="w = 0 is dust-like (no pressure); w = 1/3 is radiation")
        tb = st.slider("Bounce duration  (arbitrary units)", 0.2, 2.0, 0.6, 0.05, key="btb")
        frame_t = st.slider("Time  (negative = contracting)", -5.0, 5.0, -2.5, 0.1, key="bt")
    tt = np.linspace(-5, 5, 501)
    p = 1 / (3 * (1 + w))
    a_t = (1 + (tt / tb) ** 2) ** p
    rho_t = a_t ** (-3 * (1 + w))
    a_now = (1 + (frame_t / tb) ** 2) ** p
    rho_now = a_now ** (-3 * (1 + w))
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tt, y=a_t, mode="lines", name="scale factor a(t)  (size of universe)",
                                 line=dict(color=C["cyan"], width=3.5)))
        fig.add_trace(go.Scatter(x=tt, y=rho_t * a_t.max(), mode="lines", name="density (scaled)",
                                 line=dict(color=C["violet"], width=2.5, dash="dot"),
                                 fill="tozeroy", fillcolor=hex_alpha(C["violet"], 0.10)))
        fig.add_trace(go.Scatter(x=[frame_t], y=[a_now], mode="markers", name="now in the animation",
                                 marker=dict(size=18, color=C["aqua"], symbol="circle",
                                             line=dict(color="white", width=2))))
        fig.add_vline(x=0, line=dict(color=C["lavender"], dash="dash"),
                      annotation_text="bounce point: maximum density", annotation_font_color=C["lavender"])
        fig.add_vrect(x0=-tb, x1=tb, fillcolor=hex_alpha(C["aqua"], 0.10), line_width=0,
                      annotation_text="PBH formation zone", annotation_position="bottom",
                      annotation_font_color=C["aqua"])
        style_fig(fig, 420, xtitle="time", ytitle="relative size / density")
        fig.update_layout(legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")

    phase = "🔻 Contracting" if frame_t < -tb else ("🔺 Expanding" if frame_t > tb else "💥 At the bounce")
    a, b, c = st.columns(3)
    with a:
        card("Phase", phase, "slide the time to move through the bounce")
    with b:
        card("Relative size of universe", f"{a_now:.2f}", "1.00 = smallest (at the bounce)")
    with c:
        card("Relative density", f"{rho_now:.2f}", "1.00 = maximum density")

    st.markdown("### ⚖️ Why collapse is “inevitable” in a bounce")
    note("In the radiation-filled universe of standard inflation, <b>pressure fights collapse</b>, so only "
         "very rare ripples (far out in the tail of the distribution) become black holes. The chance falls "
         "<i>exponentially</i>. In a pressure-less (dust-like) contracting phase, nothing resists gravity, "
         "and the chance falls only as a <i>power law</i> (β ≈ 0.056 σ⁵). This is a simplified version of "
         "the argument behind Section 2.2.")
    sig = np.logspace(-3, -0.3, 300)
    dc = st.slider("Threshold for the radiation case  δc", 0.40, 0.60, 0.45, 0.01, key="bdc")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sig, y=np.clip(beta_radiation(sig, dc), 1e-300, 1), mode="lines",
                             name="radiation era (inflation): exponential", line=dict(color=C["violet"], width=3)))
    fig.add_trace(go.Scatter(x=sig, y=beta_dust(sig), mode="lines",
                             name="dust-like contraction (bounce): power law", line=dict(color=C["aqua"], width=3)))
    style_fig(fig, 380, True, True, "size of density ripples σ", "fraction collapsing into PBHs, β")
    fig.update_yaxes(range=[-60, 0])
    fig.update_layout(legend=dict(orientation="h", y=-0.28))
    st.plotly_chart(fig, width="stretch")
    s_demo = 0.02
    note(f"Example: with ripples of σ = {s_demo}, the bounce gives β ≈ {sci(beta_dust(s_demo))}, while the "
         f"radiation era gives β ≈ {sci(beta_radiation(s_demo, dc)) if beta_radiation(s_demo, dc) > 0 else '0 (below 10⁻³⁰⁰)'}. "
         "Same ripples, wildly different numbers of black holes.", "good")

    st.markdown("### 🆚 Side-by-side comparison (Section 2.3)")
    st.markdown(
        f"""
<table class="cmp">
<tr><th></th><th>🌌 Rapid inflation</th><th>🔄 Bouncing cosmology</th></tr>
<tr><td><b>Starting point</b></td><td>Big Bang singularity, then inflation</td><td>No singularity: contraction → bounce → expansion</td></tr>
<tr><td><b>What triggers PBHs</b></td><td>A special “peak” in the inflaton fluctuations</td><td>Extreme density at the bounce point</td></tr>
<tr><td><b>Fine-tuning</b></td><td>High: peak height must be just right</td><td>Lower: collapse is a natural outcome of contraction</td></tr>
<tr><td><b>Mass set by</b></td><td>Position of the peak (when it re-enters the horizon)</td><td>Scale / duration of the bounce</td></tr>
<tr><td><b>Gravitational-wave signal</b></td><td>A bump in the background spectrum</td><td>A power-law rise with “plateau” features</td></tr>
<tr><td><b>Covers sub-lunar window?</b></td><td>✅ Yes</td><td>✅ Yes</td></tr>
<tr><td><b>Test</b></td><td colspan="2" style="text-align:center">LISA / Einstein Telescope (GW background) + JWST chemistry + Roman / HSC microlensing</td></tr>
</table>
""",
        unsafe_allow_html=True,
    )

# ===========================================================================
# TAB 4 — Astrochemistry
# ===========================================================================
with tabs[3]:
    st.markdown("## How PBHs could change the first chemistry")
    paper(
        "Section 3 describes a <i>dual effect</i>: Hawking radiation can break apart H₂ (delaying the "
        "first stars), but the free electrons it creates can also <i>help</i> H₂ form. Extra ionisation "
        "also boosts HeH⁺, the universe’s first molecule."
    )
    note("🧪 <b>Toy model.</b> PBH radiation ionises hydrogen → free electrons → H₂ forms via "
         "H + e⁻ → H⁻ + γ, then H⁻ + H → H₂ + e⁻. But the same radiation also produces "
         "Lyman–Werner photons that destroy H₂. The first stars need an H₂ fraction of about "
         "5×10⁻⁴ to cool and collapse.")

    c1, c2 = st.columns([1, 2.3])
    with c1:
        logMc = st.slider("PBH mass  (log₁₀ g)", 14.7, 18.0, 15.5, 0.05, key="chemM")
        fsel = st.select_slider("Fraction of dark matter in PBHs", options=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0],
                                value=1e-4, format_func=lambda v: f"{v:g}", key="chemf")
        z = st.slider("Redshift of the forming mini-halo  (z)", 10, 40, 20, 1, key="chemz")
        n_gas = st.select_slider("Gas density in the halo  (cm⁻³)", options=[0.1, 0.3, 1, 3, 10, 30, 100],
                                 value=1, key="chemn")
        D = st.slider("H₂-destroying photons per ionisation", 1, 200, 30, 1, key="chemD")
        fabs = st.slider("Fraction of PBH energy absorbed by the gas", 0.01, 1.0, 0.1, 0.01, key="chemfa")

    T_gas = 1000.0
    k_Hm = 1.4e-18 * T_gas**0.928 * np.exp(-T_gas / 16200)
    alpha = 2.6e-13 * (T_gas / 1e4) ** -0.7
    x_res = 2e-4
    t_age = 5.4e17 / (1 + z) ** 1.5
    rho_dm0 = 2.25e-30
    n_b0 = 2.5e-7
    E_ion = 3 * 13.6 * 1.602e-12

    def chemistry(fpbh, m_g):
        P = 3.58e45 / m_g**2  # erg/s (photon Hawking power, simplified)
        zeta = fpbh * rho_dm0 * P * fabs / (m_g * n_b0 * E_ion)
        an = alpha * n_gas
        x_e = (-zeta + np.sqrt(zeta**2 + 4 * an * (zeta + an * x_res**2))) / (2 * an)
        k_lw = D * zeta
        fH2 = k_Hm * n_gas * x_e / (k_lw + 1 / t_age)
        return zeta, x_e, np.minimum(fH2, 0.5)

    fgrid = np.logspace(-8, 0, 300)
    _, xe_g, fH2_g = chemistry(fgrid, 10**logMc)
    _, xe0, fH2_0 = chemistry(0.0, 10**logMc)
    zeta_s, xe_s, fH2_s = chemistry(fsel, 10**logMc)

    with c2:
        fig = go.Figure()
        fig.add_hrect(y0=5e-4, y1=1, fillcolor=hex_alpha(C["aqua"], 0.10), line_width=0,
                      annotation_text="enough H₂ to cool → first stars form", annotation_font_color=C["aqua"],
                      annotation_position="top left")
        fig.add_trace(go.Scatter(x=fgrid, y=fH2_g, mode="lines", name="H₂ fraction",
                                 line=dict(color=C["cyan"], width=3.5)))
        fig.add_trace(go.Scatter(x=fgrid, y=xe_g, mode="lines", name="free-electron fraction",
                                 line=dict(color=C["violet"], width=2.5, dash="dot")))
        fig.add_hline(y=fH2_0, line=dict(color=C["lavender"], dash="dash", width=1),
                      annotation_text="H₂ with no PBHs", annotation_font_color=C["lavender"],
                      annotation_position="bottom right")
        fig.add_trace(go.Scatter(x=[fsel], y=[fH2_s], mode="markers", name="your setting",
                                 marker=dict(size=18, color=C["aqua"], symbol="star", line=dict(color="white", width=1.5))))
        style_fig(fig, 460, True, True, "fraction of dark matter in PBHs", "abundance (relative to hydrogen)")
        fig.update_yaxes(range=[-6, 0])
        fig.update_layout(legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")

    a, b, c, d = st.columns(4)
    with a:
        card("Ionisation rate from PBHs", f"{sci(zeta_s)} /s", "per hydrogen atom")
    with b:
        card("H₂ fraction", sci(fH2_s), f"{fH2_s/fH2_0:.2f} × the no-PBH value")
    with c:
        card("HeH⁺ boost", f"{xe_s/xe0:.2f} ×", "HeH⁺ forms from He + H⁺, so it tracks ionisation")
    with d:
        if fH2_s >= 5e-4:
            card("First stars", "🌟 can form", "H₂ cooling works")
        else:
            card("First stars", "⏳ delayed", "not enough H₂ to cool")

    ratio = fH2_s / fH2_0
    if ratio > 1.05:
        note("➕ At this setting, PBH ionisation <b>helps</b>: the extra electrons catalyse more H₂ than the "
             "radiation destroys. Cosmic Dawn could come <i>earlier</i>.", "good")
    elif ratio < 0.95:
        note("➖ At this setting, H₂ destruction <b>wins</b>. Cosmic Dawn would be <i>delayed</i>.", "dim")
    else:
        note("⚪ At this setting, PBHs barely change the chemistry.")
    note("💡 Notice the curve rises, then falls: that is the “dual-natured impact” in Section 3.1. "
         "Try raising the mass to 10¹⁷–10¹⁸ g. Hawking emission falls as 1/M², so PBHs <i>inside</i> the "
         "sub-lunar window barely affect chemistry through evaporation. For those masses, the stronger "
         "chemical effect in the literature comes from <b>gas accreting onto the PBHs</b> (not modelled here).",
         "dim")

# ===========================================================================
# TAB 5 — Gravitational waves
# ===========================================================================
with tabs[4]:
    st.markdown("## Listening for the formation: the gravitational-wave background")
    paper(
        "Section 4.1: the same big density ripples that make PBHs also shake spacetime, leaving a "
        "stochastic gravitational-wave background (SGWB). The bounce predicts a power-law rise with "
        "“plateaus” (Figure 3), distinct from inflation. LISA and the Einstein Telescope could tell them apart."
    )
    c1, c2 = st.columns([1, 2.3])
    with c1:
        logMg = st.slider("PBH mass  (log₁₀ g)", 15.0, 36.0, 20.0, 0.1, key="gwM")
        st.markdown("**🌌 Inflation peak**")
        logAg = st.slider("Peak height  log₁₀(A)", -3.0, -1.0, -1.7, 0.05, key="gwA")
        wg = st.slider("Peak width  Δ", 0.2, 2.0, 0.6, 0.05, key="gwW")
        st.markdown("**🔄 Bounce signal**")
        nt = st.slider("Power-law slope  n", 0.5, 4.0, 2.0, 0.1, key="gwn")
        logOb = st.slider("Plateau height  log₁₀ Ω", -14.0, -7.0, -9.5, 0.1, key="gwO")
        plat = st.slider("Plateau length  (decades)", 0.2, 2.5, 1.0, 0.1, key="gwP")

    fp = k_to_hz(mass_to_k(10**logMg))
    fgrid = np.logspace(-10, 4, 1200)
    A = 10**logAg
    om_inf = 1.6e-5 * A**2 * np.exp(-np.log(fgrid / fp) ** 2 / (wg**2 + 0.25)) * 0.8 * 2.5
    f_start = fp / 10**plat
    om_b = np.where(fgrid < f_start, 10**logOb * (fgrid / f_start) ** nt,
                    np.where(fgrid <= fp * 1.5, 10**logOb * (1 + 0.15 * np.sin(6 * np.log(fgrid / f_start))),
                             10**logOb * np.exp(-((fgrid / (fp * 1.5)) - 1) * 4)))

    detectors = {
        "Pulsar timing (NANOGrav)": (1e-9, 1e-7, 3e-9, 1e-9, 1.0, 2.0, C["lavender"]),
        "LISA (2030s)": (2e-5, 0.5, 3e-3, 1e-13, 3.0, 2.0, C["cyan"]),
        "DECIGO (proposed)": (1e-3, 50, 0.3, 1e-16, 3.0, 2.0, C["teal"]),
        "Einstein Telescope": (2, 5e3, 30, 1e-12, 3.0, 2.0, C["sky"]),
        "LIGO–Virgo (O3)": (10, 1e3, 25, 6e-9, 3.0, 2.0, C["mist"]),
    }

    def det_curve(lo, hi, f0, om0, a_, b_):
        f = np.logspace(np.log10(lo), np.log10(hi), 200)
        return f, om0 * 0.5 * ((f / f0) ** (-a_) + (f / f0) ** b_)

    with c2:
        fig = go.Figure()
        for name, (lo, hi, f0, om0, a_, b_, col) in detectors.items():
            f, om = det_curve(lo, hi, f0, om0, a_, b_)
            fig.add_trace(go.Scatter(
                x=np.concatenate([f, f[::-1]]), y=np.concatenate([om, np.full_like(om, 1e-3)]),
                fill="toself", fillcolor=hex_alpha(col, 0.10), line=dict(color=col, width=1.5),
                name=name, hoverinfo="name"))
        fig.add_trace(go.Scatter(x=fgrid, y=np.clip(om_inf, 1e-30, None), mode="lines",
                                 name="🌌 inflation-peak prediction", line=dict(color=C["aqua"], width=3.5)))
        fig.add_trace(go.Scatter(x=fgrid, y=np.clip(om_b, 1e-30, None), mode="lines",
                                 name="🔄 bounce prediction", line=dict(color=C["violet"], width=3.5, dash="dash")))
        fig.add_vline(x=fp, line=dict(color=C["ice"], dash="dot", width=1),
                      annotation_text=f"peak ≈ {sci(fp)} Hz", annotation_font_color=C["ice"])
        style_fig(fig, 520, True, True, "frequency (Hz)", "energy in gravitational waves, Ω_GW h²")
        fig.update_yaxes(range=[-20, -4])
        fig.update_xaxes(range=[-10, 4])
        fig.update_layout(legend=dict(orientation="h", y=-0.25, font=dict(size=11)))
        st.plotly_chart(fig, width="stretch")

    def who_hears(om_curve):
        hits = []
        for name, (lo, hi, f0, om0, a_, b_, col) in detectors.items():
            f, om = det_curve(lo, hi, f0, om0, a_, b_)
            sig = np.interp(np.log10(f), np.log10(fgrid), om_curve)
            if np.any(sig > om):
                hits.append(name.split(" (")[0])
        return hits

    h_inf, h_b = who_hears(om_inf), who_hears(om_b)
    a, b, c = st.columns(3)
    with a:
        card("Peak frequency", f"{sci(fp)} Hz", "set by when the ripple re-entered the horizon")
    with b:
        card("Inflation signal heard by", ", ".join(h_inf) if h_inf else "nobody yet 🙉")
    with c:
        card("Bounce signal heard by", ", ".join(h_b) if h_b else "nobody yet 🙉")
    note("🎧 Try the default 10²⁰ g: the signal lands right in <b>LISA’s</b> band, which is why the "
         "asteroid-mass window and LISA are so closely linked. Change the bounce slope and plateau to see how "
         "the <i>shape</i>, not just the height, could separate the two models.")
    st.caption("Detector curves are smoothed approximations of published sensitivity curves.")

# ===========================================================================
# TAB 6 — Microlensing
# ===========================================================================
with tabs[5]:
    st.markdown("## Catching a PBH today: microlensing")
    paper(
        "Section 4.2: surveys such as Subaru HSC and the upcoming Nancy Grace Roman Space Telescope watch "
        "millions of stars. A PBH passing in front of a star briefly bends and magnifies its light."
    )
    surveys = {
        "Subaru HSC → Andromeda (M31)": dict(D=770, R=1.0, cad=2, lam=620, x=0.02),
        "Nancy Grace Roman → Galactic bulge": dict(D=8, R=1.0, cad=15, lam=1500, x=0.5),
        "OGLE → Galactic bulge": dict(D=8, R=1.0, cad=20, lam=800, x=0.5),
        "EROS → Large Magellanic Cloud": dict(D=50, R=1.0, cad=1440, lam=700, x=0.5),
    }
    c1, c2 = st.columns([1, 2.3])
    with c1:
        sname = st.selectbox("Survey", list(surveys.keys()))
        S = surveys[sname]
        logMl = st.slider("PBH mass  (log₁₀ g)", 17.0, 34.0, 24.5, 0.1, key="mlM")
        xl = st.slider("Lens position  (fraction of the way to the star; small = in our own galaxy)", 0.01, 0.95, float(S["x"]), 0.01, key=f"mlx{sname}")
        u0 = st.slider("Closest approach  (in Einstein radii)", 0.0, 1.5, 0.2, 0.05, key="mlu")
        v_kms = st.slider("PBH speed across the sky  (km/s)", 50, 400, 220, 10, key="mlv")
        Rstar = st.slider("Source-star radius  (Sun = 1)", 0.3, 20.0, float(S["R"]), 0.1, key=f"mlR{sname}")

    G_SI, c_SI, kpc, Rsun = 6.674e-11, 2.998e8, 3.086e19, 6.957e8
    Mkg = 10**logMl / 1e3
    Ds = S["D"] * kpc
    R_E = np.sqrt(4 * G_SI * Mkg / c_SI**2 * Ds * xl * (1 - xl))
    tE = R_E / (v_kms * 1e3)
    rho = Rstar * Rsun * xl / R_E

    # finite-source light curve (uniform disk, sunflower sampling)
    N = 400
    idx = np.arange(N) + 0.5
    r = np.sqrt(idx / N)
    th = np.pi * (1 + 5**0.5) * idx
    dx, dy = r * np.cos(th), r * np.sin(th)
    span = 2.2 * max(1.0, rho + u0)
    tau = np.linspace(-span, span, 500)
    ux = tau[:, None] + rho * dx[None, :]
    uy = u0 + rho * dy[None, :]
    u = np.sqrt(ux**2 + uy**2) + 1e-9
    A_ps = (u**2 + 2) / (u * np.sqrt(u**2 + 4))
    A_fs = A_ps.mean(axis=1)
    u_c = np.sqrt(tau**2 + u0**2) + 1e-9
    A_point = (u_c**2 + 2) / (u_c * np.sqrt(u_c**2 + 4))
    t_sec = tau * tE

    def nice_time(s):
        if s < 120:
            return f"{s:.1f} s"
        if s < 7200:
            return f"{s/60:.1f} min"
        if s < 2 * 86400:
            return f"{s/3600:.1f} h"
        if s < 2 * 3.156e7:
            return f"{s/86400:.1f} days"
        return f"{s/3.156e7:.1f} years"

    unit, scale = ("hours", 3600) if tE * span > 3 * 3600 else ("minutes", 60)
    if tE * span > 5 * 86400:
        unit, scale = "days", 86400
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_sec / scale, y=A_point, mode="lines", name="tiny (point) star",
                                 line=dict(color=C["lavender"], width=2, dash="dot")))
        fig.add_trace(go.Scatter(x=t_sec / scale, y=A_fs, mode="lines", name="real (finite-size) star",
                                 line=dict(color=C["aqua"], width=3.5),
                                 fill="tozeroy", fillcolor=hex_alpha(C["aqua"], 0.10)))
        cad = S["cad"] * 60
        obs_t = np.arange(t_sec.min(), t_sec.max(), cad)
        if 2 <= len(obs_t) <= 400:
            obs_A = np.interp(obs_t, t_sec, A_fs)
            fig.add_trace(go.Scatter(x=obs_t / scale, y=obs_A, mode="markers", name="telescope snapshots",
                                     marker=dict(size=8, color=C["cyan"], line=dict(color="white", width=1))))
        fig.add_hline(y=1.34, line=dict(color=C["violet"], dash="dash", width=1),
                      annotation_text="standard detection level (34% brighter)", annotation_font_color=C["violet"])
        style_fig(fig, 440, xtitle=f"time from closest approach ({unit})", ytitle="brightness ÷ normal brightness")
        fig.update_yaxes(range=[0.95, max(1.6, min(15.0, float(max(A_fs.max(), A_point.max()))) * 1.08)])
        fig.update_layout(legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")

    peak = A_fs.max()
    above = t_sec[A_fs > 1.34]
    dur = (above.max() - above.min()) if above.size > 1 else 0.0
    n_pts = dur / (S["cad"] * 60) if dur > 0 else 0
    m_wave = 2e22 * (S["lam"] / 620)

    a, b, c, d = st.columns(4)
    with a:
        card("Einstein radius", f"{sci(R_E/1e3)} km", "the lens’s ‘sweet spot’ size")
    with b:
        card("Crossing time  t_E", nice_time(tE), "how long the event lasts")
    with c:
        card("Peak brightening", f"{(peak-1)*100:.1f}%", f"star size / Einstein radius = {rho:.2f}")
    with d:
        if peak < 1.34:
            card("Detectable?", "❌ washed out", "star too big compared with the lens")
        elif n_pts < 3:
            card("Detectable?", "⏱️ too fast", f"only {n_pts:.1f} snapshots during the event")
        else:
            card("Detectable?", "✅ yes", f"~{n_pts:.0f} snapshots above threshold")

    if 10**logMl < m_wave:
        note(f"🌊 Below about {sci(m_wave)} g, the black hole is so small that light behaves as a "
             "<b>wave</b> around it and the magnification fades further (not included here). Together with the "
             "finite-size effect above, this is why microlensing cannot probe the lightest PBHs, and why the "
             "window stays open at the low end.", "dim")
    note("🔍 Try lowering the mass with HSC selected. The light curve flattens as the star’s disk becomes bigger "
         "than the Einstein radius. Then switch to Roman: closer source stars help, but the cadence matters too.")

# ===========================================================================
# TAB 7 — Review map + quiz
# ===========================================================================
with tabs[6]:
    st.markdown("## How this simulator maps onto the review")
    st.markdown(
        """
<table class="cmp">
<tr><th>Review section</th><th>Idea</th><th>Where to explore it</th></tr>
<tr><td>Abstract & §1 Introduction</td><td>Dark-matter problem; mass window; Hawking evaporation & microlensing limits (Fig. 1)</td><td>🕳️ Mass Window</td></tr>
<tr><td>§2.1 Rapid inflation</td><td>Peaks in fluctuations → overdensities → collapse</td><td>🌌 Inflation Model</td></tr>
<tr><td>§2.2 Bouncing cosmology</td><td>Contraction → bounce point → PBH formation (Fig. 2)</td><td>🔄 Big Bounce Model</td></tr>
<tr><td>§2.3 Comparative analysis</td><td>Fine-tuning vs. natural collapse; different GW signals</td><td>🌌 fine-tuning plot · 🔄 comparison table · 🌊</td></tr>
<tr><td>§3.1 H₂ & cooling</td><td>Dual effect of PBH radiation on H₂ and Cosmic Dawn</td><td>🧪 Astrochemistry</td></tr>
<tr><td>§3.2 HeH⁺</td><td>Ionisation boosting the first molecule</td><td>🧪 HeH⁺ boost card</td></tr>
<tr><td>§4.1 SGWB</td><td>LISA / Einstein Telescope; plateau features (Fig. 3)</td><td>🌊 Gravitational Waves</td></tr>
<tr><td>§4.2 Microlensing</td><td>Subaru HSC & Roman</td><td>🔭 Microlensing</td></tr>
<tr><td>Abstract: Oort Cloud</td><td>PBHs passing through the outer Solar System</td><td>🕳️ Oort Cloud expander</td></tr>
</table>
""",
        unsafe_allow_html=True,
    )

    st.markdown("## 🧠 Quick quiz")
    quiz = [
        ("Why can’t PBHs lighter than about 10¹⁵ g be the dark matter today?",
         ["They are too bright", "They would have evaporated by Hawking radiation", "They are too heavy to move",
          "They only form after stars"], 1,
         "Lifetime grows as M³; a PBH of ~5×10¹⁴ g lives about the age of the Universe."),
        ("What makes the inflationary model “fine-tuned”?",
         ["It needs a bounce", "The peak height must sit in a very narrow range",
          "It only works for solar-mass PBHs", "It needs dark energy"], 1,
         "Because collapse probability depends exponentially on the ripple size, small changes in the peak "
         "give huge changes in PBH abundance."),
        ("Why does collapse happen more easily during a dust-like contraction?",
         ["There is no pressure to resist gravity", "It is colder", "Black holes evaporate faster",
          "Light travels faster"], 0,
         "With no pressure, the formation probability falls only as a power law (β ∝ σ⁵), not exponentially."),
        ("PBH radiation can both destroy and help form H₂. What helps H₂ form?",
         ["Extra free electrons from ionisation", "Higher dark-matter density", "Gravitational waves",
          "Microlensing"], 0,
         "Free electrons catalyse H₂ through the H⁻ channel."),
        ("Why does microlensing struggle with very light PBHs?",
         ["They move too slowly", "The source star looks bigger than the Einstein radius, and wave effects appear",
          "They are too far away", "They emit too much light"], 1,
         "Finite-source size and wave optics both wash out the magnification."),
    ]
    score = 0
    answered = 0
    for i, (q, opts, ans, why) in enumerate(quiz):
        st.markdown(f"**{i+1}. {q}**")
        choice = st.radio(q, opts, index=None, key=f"q{i}", label_visibility="collapsed")
        if choice is not None:
            answered += 1
            if opts.index(choice) == ans:
                score += 1
                note(f"✅ Correct! {why}", "good")
            else:
                note(f"🤔 Not quite. {why}", "dim")
    if answered:
        card("Score", f"{score} / {len(quiz)}", f"{answered} answered")

    st.markdown("## 📚 References used in the review")
    st.markdown(
        """
- Carr, B., & Kühnel, F. (2020). *Primordial Black Holes as Dark Matter: Recent Developments.* Annu. Rev. Nucl. Part. Sci., 70, 355–394.
- Franciolini, G., Musco, I., Pani, P., & Urbano, A. (2026). *The Inflationary Origin of Sub-Lunar Primordial Black Holes.* JCAP.
- Galli, S., & Paoletti, D. (2025). *Chemical Signatures of Primordial Black Hole Accretion in the Early Universe.* A&A.
- Green, A. M., & Kavanagh, B. J. (2021). *Primordial Black Holes as a Dark Matter Candidate.* J. Phys. G, 48(4), 043001.
- Inman, D., & Ali-Haïmoud, Y. (2019). *Early structure formation in primordial black hole cosmologies.* Phys. Rev. D, 100, 083528.
- Sasaki, M., Suyama, T., Tanaka, T., & Yokoyama, S. (2018). *Primordial Black Holes—Perspectives in Gravitational Wave Astronomy.* Class. Quantum Grav., 35, 063001.
- Stacy, A., & Bromm, V. (2024). *Impact of Primordial Black Hole Radiation on First Star Formation and H₂ Chemistry.* MNRAS.
- Ye, Z., Jiang, Y., & Brandenberger, R. (2026). *Bouncing Cosmology and the Stochastic Gravitational Wave Background.* PRL.
"""
    )
    st.caption("Built with Streamlit + Plotly · simplified educational models · 🌌")
