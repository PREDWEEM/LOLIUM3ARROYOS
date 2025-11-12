st.set_page_config(page_title="PREDWEEM — P2F", layout="wide")
st.title("🌾 PREDWEEM — Predicción de patrón y curva (P2F)")

bundle_file = st.sidebar.file_uploader("Bundle P2F (.joblib)", type=["joblib"])
meteo_file  = st.sidebar.file_uploader("Meteo nueva (xlsx)", type=["xlsx","xls"])
btn = st.sidebar.button("🔮 Predecir")

def longest_run(x):
    c=m=0
    for v in x: c = c+1 if v==1 else 0; m=max(m,c)
    return m

def features_from_meteo(dfm):
    dfm=standardize_cols(dfm)
    if "jd" not in dfm.columns: dfm["jd"]=np.arange(1,len(dfm)+1)
    dfm=(dfm.set_index("jd").reindex(range(1,JD_MAX+1)).interpolate().ffill().bfill().reset_index())
    tmin=dfm["tmin"].to_numpy(float); tmax=dfm["tmax"].to_numpy(float)
    tmed=(tmin+tmax)/2.0; prec=dfm["prec"].to_numpy(float); jd=dfm["jd"].to_numpy(int)
    mask=(jd>=32)&(jd<=151)
    gdd5=np.cumsum(np.maximum(tmed-5,0)); gdd3=np.cumsum(np.maximum(tmed-3,0))
    f={}
    f["gdd5_FM"]=gdd5[mask].ptp(); f["gdd3_FM"]=gdd3[mask].ptp()
    pf=prec[mask]; f["pp_FM"]=pf.sum(); f["ev10_FM"]=int((pf>=10).sum()); f["ev20_FM"]=int((pf>=20).sum())
    dry=(pf<1).astype(int); wet=(pf>=5).astype(int)
    f["dry_run_FM"]=longest_run(dry); f["wet_run_FM"]=longest_run(wet)
    def ma(x,w): import numpy as _np; k=_np.ones(w)/w; return _np.convolve(x,k,'same')
    f["tmed14_May"]=ma(tmed,14)[151]; f["tmed28_May"]=ma(tmed,28)[151]
    f["gdd5_120"]=gdd5[119]; f["pp_120"]=prec[:120].sum()
    return dfm, f

if btn:
    if not (bundle_file and meteo_file):
        st.error("Cargá el bundle P2F y la meteo."); st.stop()
    B = joblib.load(bundle_file)
    xsc=B["xsc"]; feat_names=B["feat_names"]; clf=B["clf"]; regs=B["regs"]; protos=B["protos"]

    dfm = pd.read_excel(meteo_file)
    dfm, f = features_from_meteo(dfm)
    X = np.array([[f[k] for k in feat_names]], float)
    Xs = xsc.transform(X)

    # 1) Patrón
    proba = clf.predict_proba(Xs)[0]; classes=clf.classes_
    pat = classes[np.argmax(proba)]

    # 2) Parámetros y curva
    regs_pat = regs[pat]
    p = np.array([r.predict(Xs)[0] for r in regs_pat])
    t = np.arange(1, JD_MAX+1, dtype=float)
    y = two_stage_richards(t, p)
    # Mezcla con prototipo (20%) para robustez
    y = 0.8*y + 0.2*np.clip(np.maximum.accumulate(protos[pat][:JD_MAX]),0,1)

    # Plot
    fig,ax=plt.subplots(figsize=(10,5))
    ax.plot(t, y, lw=2.5, label=f"Curva P2F (patrón {pat}, conf {proba.max():.2f})")
    ax.plot(t, protos[pat][:JD_MAX], lw=1.5, alpha=0.5, label=f"Prototipo {pat}")
    ax.set_xlim(1,JD_MAX); ax.set_ylim(0,1.02)
    ax.set_xlabel("Día juliano (1–274)"); ax.set_ylabel("Emergencia acumulada (0–1)")
    ax.grid(True, alpha=.3); ax.legend(loc="lower right")
    st.pyplot(fig)

    st.success(f"Patrón predicho: **{pat}** — conf: **{proba.max():.2f}**")

