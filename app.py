import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import plotly.express as px
 
st.set_page_config(page_title="Donor Segmentation", layout="wide")
st.title("Donor Segmentation & Prediction")
 
# ── Load & train ──────────────────────────────────────────────────────────────
@st.cache_resource
def setup():
    df = pd.read_csv("donordataset.csv").dropna(
        subset=["price", "quantity", "teacher_number_of_previously_posted_projects"]
    )
    features = ["price", "quantity", "teacher_number_of_previously_posted_projects"]
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["Segment"] = kmeans.fit_predict(X_scaled)
    df["Segment"] = df["Segment"].map({0: "Low", 1: "Mid", 2: "High"})
    reg = LinearRegression().fit(X, df["price"])
    return df, scaler, kmeans, reg, features
 
df, scaler, kmeans, reg, features = setup()
 
# ── Sidebar inputs ────────────────────────────────────────────────────────────
st.sidebar.header("Enter Donor Details")
price      = st.sidebar.number_input("Project Price ($)", 0.0, value=150.0)
quantity   = st.sidebar.number_input("Quantity", 0, value=5)
prev_posts = st.sidebar.number_input("Previous Projects Posted", 0, value=3)
 
inp        = np.array([[price, quantity, prev_posts]])
inp_scaled = scaler.transform(inp)
 
# ── KPIs ──────────────────────────────────────────────────────────────────────
k1, k2, k3 = st.columns(3)
k1.metric("Total Records",    f"{len(df):,}")
k2.metric("Avg. Price",       f"${df['price'].mean():,.2f}")
k3.metric("Avg. Quantity",    f"{df['quantity'].mean():.1f}")
 
st.divider()
 
# ── Predict buttons ───────────────────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    if st.button("Find Segment", use_container_width=True):
        seg_id   = kmeans.predict(inp_scaled)[0]
        seg_name = {0: "Low", 1: "Mid", 2: "High"}[seg_id]
        st.success(f"Segment: **{seg_name}** (Cluster {seg_id})")
with c2:
    if st.button("Predict Donation Value", use_container_width=True):
        pred = reg.predict(inp)[0]
        st.info(f"Predicted Value: **${pred:,.2f}**")
 
st.divider()
 
# ── Charts ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
 
with col1:
    st.subheader("Price by Segment")
    fig = px.box(df, x="Segment", y="price", color="Segment",
                 color_discrete_sequence=["#4ecdc4", "#f7b731", "#ff6b6b"],
                 labels={"price": "Price ($)"})
    fig.update_layout(showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)
 
with col2:
    st.subheader("Price vs Quantity")
    fig2 = px.scatter(df.sample(min(1000, len(df)), random_state=1),
                      x="quantity", y="price", color="Segment",
                      color_discrete_sequence=["#4ecdc4", "#f7b731", "#ff6b6b"],
                      opacity=0.6, labels={"price": "Price ($)", "quantity": "Quantity"})
    fig2.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(fig2, use_container_width=True)