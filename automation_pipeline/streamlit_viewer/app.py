import streamlit as st
from google.cloud import bigquery
import pandas as pd
import json, re, os, importlib.util, matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ────────────────────────────────
# Configuration
# ────────────────────────────────
PROJECT_ID = "extreme-torch-467913-m6"
BQ_TABLE = f"{PROJECT_ID}.aml_prod.df_subnetwork_summary"
os.environ["GOOGLE_CLOUD_PROJECT"] = PROJECT_ID

# ────────────────────────────────
# Helper functions — address parsing
# ────────────────────────────────
def _parse_address_list(item):
    """Parse string/list encodings of addresses safely."""
    if item is None:
        return []
    if isinstance(item, (list, tuple, set)):
        return [str(x).strip() for x in item if str(x).strip()]
    if isinstance(item, str):
        s = item.strip()
        if not s:
            return []
        # Try JSON / Python list first
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            try:
                parsed = json.loads(s.replace("'", '"'))
                if isinstance(parsed, (list, tuple, set)):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                try:
                    import ast
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, (list, tuple, set)):
                        return [str(x).strip() for x in parsed if str(x).strip()]
                except Exception:
                    pass
        # Fallback delimiter split
        parts = re.split(r"[;,|]+", s)
        return [p.strip() for p in parts if p.strip()]
    parts = re.split(r"[;,|]+", str(item).strip())
    return [p.strip() for p in parts if p.strip()]

def unique_address_count(series: pd.Series) -> int:
    """Flatten any list/string-encoded addresses and count unique entries."""
    addr_set = set()
    for item in series.dropna():
        for addr in _parse_address_list(item):
            addr_set.add(addr)
    return len(addr_set)

# ────────────────────────────────
# Fallback graph drawers
# ────────────────────────────────
def draw_single_txn_fallback(df_rank, sid: int):
    """Draw a single-node txn graph when the subnetwork has no edges."""
    ranks_sub = df_rank[df_rank["merged_subnetwork_id"] == sid]
    if ranks_sub.empty:
        return
    node_id = str(ranks_sub["node"].iloc[0])
    G = nx.DiGraph()
    G.add_node(node_id)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    pos = {node_id: (0.0, 0.0)}
    nx.draw_networkx_nodes(G, pos, nodelist=[node_id], node_size=500,
                           node_color="#FFD580", edgecolors="#FF8C00",
                           linewidths=1.4, alpha=0.95, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={node_id: f"TxnID: {node_id}"}, font_size=9, ax=ax)
    ax.set_title(f"Txn→Txn Directed Graph for Subnetwork {sid} (single node)")
    ax.axis("off")
    return G

def draw_addr_fallback(df_addrtxn, df_txnaddr, df_rank, sid: int):
    """Draw addresses as isolated nodes when no addr→addr edges can be projected."""
    ranks_sub = df_rank[df_rank["merged_subnetwork_id"] == sid]
    if ranks_sub.empty:
        return
    illicit_txns = set(ranks_sub["node"].astype(str))
    ain = df_addrtxn[df_addrtxn["txId"].astype(str).isin(illicit_txns)]
    aout = df_txnaddr[df_txnaddr["txId"].astype(str).isin(illicit_txns)]
    addrs = set(ain["input_address"].dropna().astype(str)) | set(aout["output_address"].dropna().astype(str))
    if not addrs:
        return
    G = nx.DiGraph()
    G.add_nodes_from(addrs)
    nodes = list(addrs)
    theta = np.linspace(0, 2 * np.pi, num=len(nodes), endpoint=False)
    pos = {n: (float(np.cos(t)), float(np.sin(t))) for n, t in zip(nodes, theta)}
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=300,
                           node_color="#FFD580", edgecolors="#FF8C00",
                           linewidths=1.2, alpha=0.95, ax=ax)
    to_label = set(nodes[: min(10, len(nodes))])
    nx.draw_networkx_labels(G, pos, labels={n: n for n in to_label}, font_size=7, ax=ax)
    ax.set_title(f"Addr→Addr Graph for Subnetwork {sid} (no edges; showing addresses)")
    ax.axis("off")
    return G

# ────────────────────────────────
# BigQuery Client
# ────────────────────────────────
client = bigquery.Client(project=PROJECT_ID)

@st.cache_data(show_spinner=False)
def load_summary_table() -> pd.DataFrame:
    query = f"SELECT * FROM `{BQ_TABLE}`"
    return client.query(query).to_dataframe()

df_all = load_summary_table()

@st.cache_data(show_spinner=True)
def load_graph_data():
    """Load and cache tables needed for visualisation."""
    df_rank = client.query(f"SELECT * FROM `{PROJECT_ID}.aml_prod.df_rank_composite`").to_dataframe()
    illicit_ids = set(df_rank["node"].astype(str))
    df_network_edges = client.query(f"SELECT * FROM `{PROJECT_ID}.aml_prod.df_network_edges`").to_dataframe()
    df_addrtxn = client.query(f"SELECT * FROM `{PROJECT_ID}.actor.addrtxn_edgelist`").to_dataframe()
    df_addrtxn["txId"] = df_addrtxn["txId"].astype(str)
    df_addrtxn = df_addrtxn[df_addrtxn["txId"].isin(illicit_ids)]
    df_txnaddr = client.query(f"SELECT * FROM `{PROJECT_ID}.actor.txaddr_edgelist`").to_dataframe()
    df_txnaddr["txId"] = df_txnaddr["txId"].astype(str)
    df_txnaddr = df_txnaddr[df_txnaddr["txId"].isin(illicit_ids)]
    return df_rank, df_network_edges, df_addrtxn, df_txnaddr

# ────────────────────────────────
# Page Layout
# ────────────────────────────────
st.set_page_config(page_title="Bitcoin AML Network Viewer", layout="wide")

st.markdown("<h2 style='text-align: center; margin-bottom: 0;'>Bitcoin AML Network Viewer</h2>", unsafe_allow_html=True)
st.markdown(
    """
    <p style='text-align: center; color: grey; font-size: 16px;'>
    This dashboard helps investigators prioritise potential money-laundering networks detected on the Bitcoin blockchain. 
    Each subnetwork represents a cluster of transactions and addresses that have been algorithmically identified and ranked 
    based on suspicious behavioural patterns. The goal is to support efficient triage and evidence-based investigation.
    </p>
    """,
    unsafe_allow_html=True,
)

# ────────────────────────────────
# Global Quick Stats
# ────────────────────────────────
if df_all.empty:
    st.warning("No data found in df_subnetwork_summary.")
    st.stop()

global_illicit_networks = df_all["subnetwork_id"].nunique()
global_illicit_txns = df_all["txn_id"].nunique()
global_unique_senders = unique_address_count(df_all["input_addresses"])
global_unique_receivers = unique_address_count(df_all["output_addresses"])
global_timesteps = "25"

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Global: Unique Illicit Networks", f"{global_illicit_networks:,}")
col2.metric("Global: Number of Timesteps", global_timesteps)
col3.metric("Global: Unique Senders", f"{global_unique_senders:,}")
col4.metric("Global: Unique Receivers", f"{global_unique_receivers:,}")
col5.metric("Global: Illicit Transactions", f"{global_illicit_txns:,}")

st.markdown("---")

# ────────────────────────────────
# Sidebar selection (auto-loads; no button)
# ────────────────────────────────
with st.sidebar:
    st.header("Search Subnetwork")
    st.caption(
        "Use the dropdown to select a **Subnetwork ID**. The dashboard automatically loads the "
        "summary table and both network visualisations (Txn→Txn and Addr→Addr) for the selected subnetwork."
    )
    subnet_options = sorted(df_all["subnetwork_id"].unique())
    if not subnet_options:
        st.error("No subnetworks available.")
        st.stop()
    sid = st.selectbox("Select Subnetwork ID", subnet_options, index=0)

# ────────────────────────────────
# Query selected subnetwork
# ────────────────────────────────
@st.cache_data(show_spinner=False)
def get_summary(subnetwork_id: int) -> pd.DataFrame:
    query = f"SELECT * FROM `{BQ_TABLE}` WHERE subnetwork_id = @sid"
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("sid", "INT64", int(subnetwork_id))]
    )
    return client.query(query, job_config=job_config).to_dataframe()

# ────────────────────────────────
# Filtered Subnetwork Section (auto-loads on selection)
# ────────────────────────────────
if sid is not None:
    with st.spinner("Querying BigQuery..."):
        df = get_summary(int(sid))

    if df.empty:
        st.warning(f"No summary found for subnetwork_id {sid}")
    else:
        st.markdown(
            f"""
            <h3 style='text-align: center;'>Filtered Subnetwork Analysis — Subnetwork {sid}</h3>
            <p style='text-align: center; color: grey; font-size: 15px;'>
            These results focus on the selected subnetwork, summarising its key transaction features 
            and participants to guide further review.
            </p>
            """,
            unsafe_allow_html=True,
        )

        # Quick stats
        subnet_txn_count = df["txn_id"].nunique() if "txn_id" in df.columns else len(df)
        subnet_timestamp = "25"
        subnet_unique_senders = unique_address_count(df["input_addresses"])
        subnet_unique_receivers = unique_address_count(df["output_addresses"])

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Transactions in Subnetwork", f"{subnet_txn_count:,}")
        s2.metric("Timestamp", subnet_timestamp)
        s3.metric("Unique Senders", f"{subnet_unique_senders:,}")
        s4.metric("Unique Receivers", f"{subnet_unique_receivers:,}")

        # Sort and table
        if "investigation_order" in df.columns:
            df = df.sort_values("investigation_order", ascending=True).reset_index(drop=True)
        cols_to_show = [
            "subnetwork_id","txn_id","investigation_order","in_BTC_total",
            "n_inputs","n_outputs","input_addresses","output_addresses"
        ]
        available_cols = [c for c in cols_to_show if c in df.columns]
        df = df[available_cols].rename(columns={
            "subnetwork_id":"Subnetwork ID","txn_id":"Transaction ID",
            "investigation_order":"Investigation Order","in_BTC_total":"Inbound BTC (to Txn)",
            "n_inputs":"No. Sender Addresses","n_outputs":"No. Receiver Addresses",
            "input_addresses":"Sender Addresses","output_addresses":"Receiver Addresses"
        })
        for col in ["No. Sender Addresses","No. Receiver Addresses","Inbound BTC (to Txn)"]:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
        if "Inbound BTC (to Txn)" in df.columns:
            df["Inbound BTC (to Txn)"] = df["Inbound BTC (to Txn)"].map("{:,.2f}".format)

        st.caption("Data source: BigQuery table `aml_prod.df_subnetwork_summary` — updated via Vertex AI pipeline.")
        st.markdown(f"<h4>Investigation Priority Table — Subnetwork {sid}</h4>", unsafe_allow_html=True)
        if "Investigation Order" in df.columns:
            st.dataframe(df.style.background_gradient(subset=["Investigation Order"], cmap="YlOrRd_r"),
                         use_container_width=True, height=350)
        else:
            st.dataframe(df, use_container_width=True, height=350)

        # ────────────────────────────────
        # Network Visualisations (stacked)
        # ────────────────────────────────
        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align:center;'>Network Visualisations — Subnetwork {sid}</h4>", unsafe_allow_html=True)

        # Load txn_subnetworks module (local)
        module_path = os.path.join(os.path.dirname(__file__), "txn_subnetworks.py")
        spec = importlib.util.spec_from_file_location("txn_subnetworks", module_path)
        txn_subnetworks = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(txn_subnetworks)
        build_vis = txn_subnetworks.visualise_subnetwork()

        # Cached graph data
        df_rank, df_network_edges, df_addrtxn, df_txnaddr = load_graph_data()

        # Txn→Txn graph
        st.markdown("<h5 style='text-align:center;'>Txn→Txn Subnetwork</h5>", unsafe_allow_html=True)
        with st.spinner("Rendering Txn→Txn subnetwork..."):
            plt.close("all")
            G_txn = build_vis.plot_txntxn_subnetwork(
                df_network_edges, df_rank,
                subnetwork_id=int(sid),
                size_by="pagerank", label_by="composite_rank", top_n_label=5
            )
            fig1 = plt.gcf()
            if (G_txn is None) or (len(getattr(G_txn, "nodes", [])) == 0):
                plt.close("all")
                G_txn = draw_single_txn_fallback(df_rank, int(sid))
                fig1 = plt.gcf()
            st.pyplot(fig1)
            if G_txn is not None:
                st.caption(f"Txn–Txn network: {len(G_txn.nodes)} nodes, {len(G_txn.edges)} edges")

        # Addr→Addr graph
        st.markdown("<h5 style='text-align:center;'>Addr→Addr Subnetwork</h5>", unsafe_allow_html=True)
        with st.spinner("Rendering Addr→Addr subnetwork..."):
            plt.close("all")
            G_addr = build_vis.plot_addraddr_subnetwork(
                df_addrtxn, df_txnaddr, df_rank, df_network_edges,
                merged_subnetwork_id=int(sid),
                uniform_size=False, others_size=100, top_n_label=5, k=1.1
            )
            fig2 = plt.gcf()
            if (G_addr is None) or (len(getattr(G_addr, "nodes", [])) == 0):
                plt.close("all")
                G_addr = draw_addr_fallback(df_addrtxn, df_txnaddr, df_rank, int(sid))
                fig2 = plt.gcf()
            st.pyplot(fig2)
            if G_addr is not None:
                st.caption(f"Addr–Addr network: {len(G_addr.nodes)} nodes, {len(G_addr.edges)} edges")

# ────────────────────────────────
# Footer
# ────────────────────────────────
st.markdown(
    """
    <div id='custom-footer'>
      Jennifer Payne — Master of Data Science, RMIT University<br>
      <i>Minor Thesis: Detection, Ranking and Visualisation of Money Laundering Networks on the Bitcoin Blockchain (2025)</i><br><br>
      <b>Code and Data Availability:</b><br>
      All scripts, trained models, and data processing notebooks developed for this research are available at 
      <a class='footer-link' href='https://github.com/majorpayne-2021/rmit_master_thesis' target='_blank'>
      github.com/majorpayne-2021/rmit_master_thesis</a>.<br>
      The Elliptic++ dataset is available at 
      <a class='footer-link' href='https://github.com/git-disl/EllipticPlusPlus' target='_blank'>
      github.com/git-disl/EllipticPlusPlus</a>.
    </div>
    """,
    unsafe_allow_html=True,
)
