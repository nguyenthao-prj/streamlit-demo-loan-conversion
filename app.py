import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from preprocessing.score_cnn import score_dataframe

st.set_page_config(page_title="Loan Conversion", layout="wide")

PREPROCESS_PATH = "artifacts/preprocess.pkl"

@st.cache_resource
def load_preprocessor():
    return joblib.load(PREPROCESS_PATH)

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def compute_dti(df: pd.DataFrame) -> pd.Series:
    # DTI (%) = Monthly Debt * 12 / Annual Income * 100
    inc = df.get("Annual Income", pd.Series([np.nan]*len(df))).apply(safe_float)
    debt = df.get("Monthly Debt", pd.Series([np.nan]*len(df))).apply(safe_float)
    dti = (debt * 12 / inc) * 100
    dti = dti.replace([np.inf, -np.inf], np.nan)
    return dti

# ----------------------------
# Header
# ----------------------------
st.title("Priority Customers for Loan Conversion")
st.caption("Decision dashboard for short → long-term loan conversion (Top k% based on model score).")
st.divider()

# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload customer snapshot (CSV)", type=["csv"])

    top_k = st.slider(
        "Select priority threshold (Top k%)",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )
    show_priority_only = st.checkbox("Show only priority customers", value=False)

if uploaded_file is None:
    st.info("⬅️ Upload a CSV file to start.")
    st.stop()

# ----------------------------
# Load input
# ----------------------------
try:
    df_raw = pd.read_csv(uploaded_file)
except Exception as e:
    st.error("Cannot read CSV. Please check file format/encoding.")
    st.exception(e)
    st.stop()


# ----------------------------
# SCORE (must happen before dashboard merge/KPIs/charts)
# ----------------------------
results = score_dataframe(df_raw, top_k=top_k)


# Preprocess for profile metrics (NOT for model input)
pre = load_preprocessor()
df_business = df_raw.copy()
if "Customer ID" in df_business.columns:
    df_business["Customer ID"] = df_business["Customer ID"].astype(str).str.strip()

results["Customer ID"] = results["Customer ID"].astype(str).str.strip()

df_business = df_business.merge(
    results[["Customer ID", "Conversion Probability (%)", "Priority Group"]],
    on="Customer ID",
    how="left"
).copy()

top_label = f"Top {top_k}%"
df_business["Priority Group"] = df_business["Priority Group"].fillna("Remaining")
df_business["Conversion Probability (%)"] = pd.to_numeric(
    df_business["Conversion Probability (%)"], errors="coerce"
)

df_business["DTI (%)"] = compute_dti(df_business)
df_dash = df_business.copy()

# ----------------------------
# KPI SUMMARY (business)
# ----------------------------
total_n = len(df_dash)
top_mask = df_dash["Priority Group"] == top_label
top_n = int(top_mask.sum())

# Existing-data KPIs
loan_amt_top = df_dash.loc[top_mask, "Current Loan Amount"].apply(safe_float).sum()
avg_cs_top = df_dash.loc[top_mask, "Credit Score"].apply(safe_float).mean()
avg_dti_top = df_dash.loc[top_mask, "DTI (%)"].apply(safe_float).mean()

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Customers Scored", f"{total_n:,}")
k2.metric(top_label + " Customers", f"{top_n:,}")
k3.metric("Coverage", f"{top_k:.1f}%")
k4.metric("Total Loan Amount (Top k%)", f"{loan_amt_top:,.0f}")
k5.metric("Avg Credit Score (Top k%)", f"{avg_cs_top:,.0f}" if not np.isnan(avg_cs_top) else "—")
k6.metric("Avg DTI % (Top k%)", f"{avg_dti_top:,.1f}%" if not np.isnan(avg_dti_top) else "—")

st.divider()

# ----------------------------
# CHARTS (Decision-first layout)
# ----------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("Nhóm khách hàng ưu tiên đang đến từ kỳ hạn vay nào?")

    # Prefer encoded column created by preprocessor
    if "Term_Long Term" in df_dash.columns:
        term_series = np.where(
            df_dash["Term_Long Term"].apply(safe_float).fillna(0).astype(int) == 1,
            "Long Term",
            "Short Term"
        )
        term_counts = pd.Series(term_series)[top_mask].value_counts()
        st.bar_chart(term_counts)

    # Fallback if raw Term exists (rare)
    elif "Term" in df_dash.columns:
        term_counts = df_dash.loc[top_mask, "Term"].astype(str).value_counts()
        st.bar_chart(term_counts)

    else:
        st.info("Không tìm thấy thông tin kỳ hạn vay (Term / Term_Long Term).")


with right:
    st.subheader("Khách hàng nào vừa có năng lực tín dụng, vừa có động cơ chuyển đổi?")


    plot_df = df_dash.copy()
    plot_df["Credit Score"] = plot_df["Credit Score"].apply(safe_float)
    plot_df["DTI (%)"] = plot_df["DTI (%)"].apply(safe_float)

    # Filter to business-valid ranges
    plot_df = plot_df[
        plot_df["Credit Score"].between(300, 850, inclusive="both") &
        plot_df["DTI (%)"].between(0, 200, inclusive="both")
    ].copy()

    # ---- Rule-of-thumb zone (tunable thresholds) ----
    CS_MIN = 680
    DTI_MIN = 10
    DTI_MAX = 25
    # % of Top k% inside rule-of-thumb zone
    in_zone = (
        (plot_df["Credit Score"] >= CS_MIN) &
        (plot_df["DTI (%)"] >= DTI_MIN) &
        (plot_df["DTI (%)"] <= DTI_MAX)
    )

    top_only = plot_df["Priority Group"] == top_label
    top_in_zone_pct = (in_zone & top_only).sum() / max(top_only.sum(), 1) * 100
    overall_in_zone_pct = in_zone.sum() / max(len(plot_df), 1) * 100

    mA, mB = st.columns(2)
    mA.metric(
    f"Tỷ lệ KH ưu tiên trong nhóm Top {top_k}% ",
    f"{top_in_zone_pct:.1f}%"
    )

    mB.metric(
    "Số KH phù hợp trong tổng số khách hàng",
    f"{overall_in_zone_pct:.1f}%"
    )

    fig2 = plt.figure()
    ax = plt.gca()

    # Shade sweet-spot area
    ax.axvspan(CS_MIN, 850, alpha=0.08)
    ax.axhspan(DTI_MIN, DTI_MAX, alpha=0.06)
    ax.add_patch(
        plt.Rectangle(
            (CS_MIN, DTI_MIN),
            850 - CS_MIN,
            DTI_MAX - DTI_MIN,
            fill=True,
            alpha=0.10
        )
    )
    ax.text(
        CS_MIN + 5,
        DTI_MAX + 1,
        "Rule-of-thumb zone\n(CS ≥ 680, DTI 10–25%)",
        fontsize=10
    )

    # Scatter points by group
    for g in [top_label, "Remaining"]:
        sub = plot_df[plot_df["Priority Group"] == g]
        ax.scatter(sub["Credit Score"], sub["DTI (%)"], label=g, alpha=0.7, s=20)

    ax.set_xlabel("Credit Score (raw)")
    ax.set_ylabel("DTI (%) (raw)")
    ax.legend()
    st.pyplot(fig2)

# ----------------------------
# CHART 4 & 5: Loan Amount vs Income
# ----------------------------
st.divider()
c4, c5 = st.columns(2)

# ---------- Chart 4 ----------
with c4:
    st.subheader(
        "Nhóm ưu tiên có phân bố quy mô khoản vay khác gì so với phần còn lại?"
    )

    loan_df = df_dash.copy()
    loan_df["Current Loan Amount"] = loan_df["Current Loan Amount"].apply(safe_float)
    loan_df = loan_df.dropna(subset=["Current Loan Amount"])

    top_vals = loan_df.loc[
        loan_df["Priority Group"] == top_label,
        "Current Loan Amount"
    ]
    rem_vals = loan_df.loc[
        loan_df["Priority Group"] == "Remaining",
        "Current Loan Amount"
    ]

    fig4 = plt.figure()
    plt.boxplot(
        [top_vals, rem_vals],
        labels=[top_label, "Remaining"],
        showfliers=False
    )
    plt.ylabel("Current Loan Amount")
    plt.grid(axis="y", alpha=0.3)
    st.pyplot(fig4)


# ---------- Chart 5 ----------
with c5:
    st.subheader(
        "Nhóm ưu tiên có nền tảng thu nhập khác gì so với phần còn lại?"
    )

    inc_df = df_dash.copy()
    inc_df["Annual Income"] = inc_df["Annual Income"].apply(safe_float)
    inc_df = inc_df.dropna(subset=["Annual Income"])

    top_inc = inc_df.loc[
        inc_df["Priority Group"] == top_label,
        "Annual Income"
    ]
    rem_inc = inc_df.loc[
        inc_df["Priority Group"] == "Remaining",
        "Annual Income"
    ]

    fig5 = plt.figure()
    plt.boxplot(
        [top_inc, rem_inc],
        labels=[top_label, "Remaining"],
        showfliers=False
    )
    plt.ylabel("Annual Income")
    plt.grid(axis="y", alpha=0.3)
    st.pyplot(fig5)


# ----------------------------
# CHART 6: Prioritization within Top k% + Call-first table
# ----------------------------
st.divider()
st.subheader("Trong nhóm ưu tiên, nên tiếp cận khách hàng nào trước?")

# Use df_dash (already has Priority Group + score + raw business vars)
c6_df = df_dash[df_dash["Priority Group"] == top_label].copy()

# Ensure numeric
c6_df["Current Loan Amount"] = c6_df.get("Current Loan Amount", np.nan).apply(safe_float)
c6_df["DTI (%)"] = c6_df.get("DTI (%)", np.nan).apply(safe_float)
c6_df["Credit Score"] = c6_df.get("Credit Score", np.nan).apply(safe_float)
c6_df["Annual Income"] = c6_df.get("Annual Income", np.nan).apply(safe_float)
c6_df["Conversion Probability (%)"] = pd.to_numeric(
    c6_df.get("Conversion Probability (%)", np.nan), errors="coerce"
)

# Optional: remove extreme placeholder loan values
c6_df = c6_df[c6_df["Current Loan Amount"].fillna(np.inf) < 90_000_000].copy()

# Drop rows missing key axes
c6_df = c6_df.dropna(subset=["Current Loan Amount", "DTI (%)"])

if len(c6_df) < 5:
    st.info("Không đủ dữ liệu để vẽ Chart 6 (Top k% quá ít hoặc thiếu Loan/DTI).")
else:
    # Medians for quadrants
    x_med = float(np.nanmedian(c6_df["Current Loan Amount"].values))
    y_med = float(np.nanmedian(c6_df["DTI (%)"].values))

    fig6 = plt.figure(figsize=(6, 4))
    ax = plt.gca()

    # Quadrant lines
    ax.axvline(x_med, linestyle="--", alpha=0.7)
    ax.axhline(y_med, linestyle="--", alpha=0.7)

    # Scatter points
    ax.scatter(
        c6_df["Current Loan Amount"],
        c6_df["DTI (%)"],
        alpha=0.85,
        s=25
    )

    ax.set_xlabel("Quy mô khoản vay (Current Loan Amount)")
    ax.set_ylabel("Áp lực nợ / thu nhập (DTI %)")

    # Quadrant labels (stable placement)
    ax.text(0.62, 0.10, "Ưu tiên gọi trước\n(Loan cao, DTI thấp)", transform=ax.transAxes, fontsize=9)
    ax.text(0.62, 0.72, "Giá trị cao\ncần thẩm định", transform=ax.transAxes, fontsize=9)
    ax.text(0.05, 0.10, "Kênh chi phí thấp\n(Loan thấp, DTI thấp)", transform=ax.transAxes, fontsize=9)
    ax.text(0.05, 0.72, "Không ưu tiên\n(Loan thấp, DTI cao)", transform=ax.transAxes, fontsize=9)

    ax.ticklabel_format(style="plain", axis="x")
    st.pyplot(fig6)

    # KPI summary
    q1 = ((c6_df["Current Loan Amount"] >= x_med) & (c6_df["DTI (%)"] < y_med)).sum()
    q2 = ((c6_df["Current Loan Amount"] >= x_med) & (c6_df["DTI (%)"] >= y_med)).sum()
    q3 = ((c6_df["Current Loan Amount"] < x_med) & (c6_df["DTI (%)"] < y_med)).sum()
    q4 = ((c6_df["Current Loan Amount"] < x_med) & (c6_df["DTI (%)"] >= y_med)).sum()

    kA, kB, kC, kD = st.columns(4)
    kA.metric("Ưu tiên gọi trước", f"{q1}")
    kB.metric("Giá trị cao cần thẩm định", f"{q2}")
    kC.metric("Kênh chi phí thấp", f"{q3}")
    kD.metric("Không ưu tiên", f"{q4}")

    # ----------------------------
    # TABLE: Priority call-first customer profiles
    # ----------------------------
    st.markdown("### Danh sách hồ sơ KH — Ưu tiên gọi trước")

    call_first = c6_df[
        (c6_df["Current Loan Amount"] >= x_med) &
        (c6_df["DTI (%)"] < y_med)
    ].copy()

    if len(call_first) == 0:
        st.info("Không có khách hàng nào nằm trong vùng 'Ưu tiên gọi trước'.")
    else:
        # Term snapshot
        if "Term (snapshot)" not in call_first.columns and "Term_Long Term" in call_first.columns:
            call_first["Term (snapshot)"] = np.where(
                call_first["Term_Long Term"].apply(safe_float).fillna(0).astype(int) == 1,
                "Long Term",
                "Short Term"
            )

        # Rename for readability
        call_first = call_first.rename(columns={
            "Conversion Probability (%)": "Priority Score (Percentile)",
            "Current Loan Amount": "Loan Amount",
        })

        # Sort: Loan high first, then score
        sort_cols = []
        if "Loan Amount" in call_first.columns:
            sort_cols.append("Loan Amount")
        if "Priority Score (Percentile)" in call_first.columns:
            sort_cols.append("Priority Score (Percentile)")

        if sort_cols:
            call_first = call_first.sort_values(
                sort_cols, ascending=[False] * len(sort_cols)
            ).reset_index(drop=True)
        else:
            call_first = call_first.reset_index(drop=True)

        call_first.insert(0, "Rank", call_first.index + 1)

        display_cols = [
            "Rank",
            "Customer ID",
            "Priority Score (Percentile)",
            "Loan Amount",
            "Credit Score",
            "Annual Income",
            "DTI (%)",
            "Term (snapshot)",
            "Purpose",
            "Home Ownership",
        ]
        display_cols = [c for c in display_cols if c in call_first.columns]

        TOP_N = 30
        st.caption(f"Hiển thị Top {min(TOP_N, len(call_first))} hồ sơ (ưu tiên theo Loan Amount và Priority Score).")
        st.dataframe(call_first[display_cols].head(TOP_N), use_container_width=True, height=420)

        csv_bytes = call_first[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Tải danh sách “Ưu tiên gọi trước” (CSV)",
            data=csv_bytes,
            file_name=f"call_first_top{top_k}pct.csv",
            mime="text/csv"
        )

st.divider()


# ----------------------------
# ACTION TABLE
# ----------------------------
st.subheader("Danh sách khách hàng ưu tiên 🔥")

table_df = results.rename(columns={"Conversion Probability (%)": "Priority Score"}).copy()

# Ensure numeric for proper sorting
table_df["Priority Score"] = pd.to_numeric(table_df["Priority Score"], errors="coerce")

# Put Top k% first, then sort by score desc within each group
table_df["__is_top"] = (table_df["Priority Group"] == top_label).astype(int)

table_df = table_df.sort_values(
    by=["__is_top", "Priority Score"],
    ascending=[False, False],
    na_position="last"
).drop(columns=["__is_top"])

# Optional: show only priority customers
if show_priority_only:
    table_df = table_df[table_df["Priority Group"] == top_label].copy()

# Re-rank after sorting
table_df = table_df.reset_index(drop=True)
table_df.insert(0, "Rank", table_df.index + 1)
table_df.insert(1, "Flag", table_df["Priority Group"].apply(lambda x: "🔥" if x == top_label else ""))

st.dataframe(
    table_df[["Rank", "Flag", "Customer ID", "Priority Group", "Priority Score"]],
    use_container_width=True,
    height=520
)

# Download top-k list
top_list = results[results["Priority Group"] == top_label].copy()
csv_bytes = top_list.to_csv(index=False).encode("utf-8")

st.download_button(
    label=f"Download {top_label} list (CSV)",
    data=csv_bytes,
    file_name=f"priority_{top_k}pct.csv",
    mime="text/csv"
)


st.caption("Priority Score is a percentile rank within the uploaded population.")
