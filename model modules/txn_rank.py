# -------------------------------------------------------------------------
# Module for Transaction Ranking and Subnetwork Analysis
# -------------------------------------------------------------------------
# This module implements all functions required to calculate, compare, and analyse
# ranking metrics within illicit subnetworks of the Bitcoin transaction graph.
# It forms the ranking and evaluation stage of the AML detection pipeline,
# quantifying node influence and identifying structurally important transactions.
#
# The module contains two main classes:
# - build_txn_rank: Provides ranking utilities for directed transaction subnetworks.
#   It includes multiple network-based algorithms such as PageRank, HITS, degree
#   metrics, and centrality measures, as well as percentile-based composite ranking
#   calculations that integrate topological and financial features.
# - ranking_comparison: Computes correlation matrices and visualisations to compare
#   ranking methods within and across subnetworks. It supports per-subnetwork
#   correlation analysis, average correlation aggregation, and heatmap visualisation
#   for ranking agreement diagnostics.
#
# Together, these classes provide a comprehensive framework for ranking and
# evaluating nodes in Bitcoin subnetworks using both classical graph theory metrics
# and custom composite ranking approaches.
#
# This code forms part of the technical work developed in support of the research
# thesis titled:
# “Detection, Ranking and Visualisation of Money Laundering Networks on the Bitcoin Blockchain”
# by Jennifer Payne (RMIT University).
#
# GitHub Repository: https://github.com/majorpayne-2021/rmit_master_thesis
# Elliptic++ Dataset Source: https://github.com/git-disl/EllipticPlusPlus
# -------------------------------------------------------------------------

import numpy as np
import pandas as pd
import networkx as nx
import math
from collections import defaultdict
from typing import Iterable, List, Optional, Dict, Tuple
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
class build_txn_rank:
# -------------------------------------------------------------------------
    """
    Ranking utilities for directed transaction graphs, evaluated per subnetwork.
    Primary outputs are node-level scores and ordinal ranks within each subnetwork.
    """

    def __init__(self):
        self.df = pd.DataFrame()

    # ---------- helpers ----------
    @staticmethod
    def _digraph(df: pd.DataFrame, src: str = "src_txn_id", dst: str = "dst_txn_id") -> nx.DiGraph:
        """Construct a directed graph from an edge list with columns [src, dst]."""
        G = nx.DiGraph()
        if src in df.columns and dst in df.columns and not df.empty:
            G.add_edges_from(df[[src, dst]].to_records(index=False))
        return G

    @staticmethod
    def _rank_in_group(
        df: pd.DataFrame, group: str, col: str, rank_col: str, ascending: bool = False
    ) -> pd.DataFrame:
        """Dense-rank `col` within `group` and write to `rank_col`."""
        if len(df) == 0:
            return df
        df[rank_col] = df.groupby(group)[col].rank(method="dense", ascending=ascending)
        return df

    # ---------- 1) PageRank ----------
    def build_pagerank(
        self,
        df_edges: pd.DataFrame,
        *,
        group: str = "merged_subnetwork_id",
        src: str = "src_txn_id",
        dst: str = "dst_txn_id",
        weight_col: Optional[str] = None,
        alpha: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-8,
        aggregate_duplicates: bool = True,
    ) -> pd.DataFrame:
        """
        Compute PageRank per subnetwork and return pagerank values with ordinal ranks.
        Returns: [group, node, pagerank, pr_rank]
        """
        if df_edges.empty:
            return pd.DataFrame(columns=[group, "node", "pagerank", "pr_rank"])

        results: List[pd.DataFrame] = []

        for gid, gdf in df_edges.groupby(group, sort=False):
            cols = [src, dst] + ([weight_col] if (weight_col and weight_col in gdf.columns) else [])
            edf = gdf[cols].dropna(subset=[src, dst])

            G = nx.DiGraph()
            if weight_col and weight_col in edf.columns:
                if aggregate_duplicates:
                    edf = edf.groupby([src, dst], as_index=False)[weight_col].sum()
                G.add_weighted_edges_from(edf[[src, dst, weight_col]].itertuples(index=False, name=None))
                nx_weight = "weight"
            else:
                if aggregate_duplicates:
                    edf = edf.drop_duplicates([src, dst])
                G.add_edges_from(edf[[src, dst]].itertuples(index=False, name=None))
                nx_weight = None

            if len(G) == 0:
                continue

            pr = nx.pagerank(G, alpha=alpha, max_iter=max_iter, tol=tol, weight=nx_weight)

            out = pd.DataFrame({group: gid, "node": list(pr.keys()), "pagerank": list(pr.values())})
            out["pr_rank"] = out.groupby(group)["pagerank"].rank(method="dense", ascending=False).astype(int)
            results.append(out[[group, "node", "pagerank", "pr_rank"]])

        if not results:
            return pd.DataFrame(columns=[group, "node", "pagerank", "pr_rank"])

        return (
            pd.concat(results, ignore_index=True)
            .sort_values([group, "pagerank"], ascending=[True, False])
            .reset_index(drop=True)
        )

    # ---------- 2) HITS (unweighted) ----------
    def build_hits(
        self,
        df_edges: pd.DataFrame,
        *,
        group: str = "merged_subnetwork_id",
        src: str = "src_txn_id",
        dst: str = "dst_txn_id",
        max_iter: int = 1000,
        normalized: bool = True,
    ) -> pd.DataFrame:
        """
        Compute HITS hub/authority scores and ranks per subnetwork (unweighted).
        Returns: [group, node, hub_score, authority_score, hub_rank, authority_rank]
        """
        if df_edges.empty:
            return pd.DataFrame(columns=[group, "node", "hub_score", "authority_score", "hub_rank", "authority_rank"])

        out: List[pd.DataFrame] = []
        for gid, ed in df_edges.groupby(group, sort=False):
            G = self._digraph(ed, src, dst)
            if len(G) == 0:
                continue
            hubs, auths = nx.hits(G, max_iter=max_iter, normalized=normalized)
            df = pd.DataFrame(
                {
                    group: gid,
                    "node": list(hubs.keys()),
                    "hub_score": list(hubs.values()),
                    "authority_score": [auths[n] for n in hubs.keys()],
                }
            )
            self._rank_in_group(df, group, "hub_score", "hub_rank", ascending=False)
            self._rank_in_group(df, group, "authority_score", "authority_rank", ascending=False)
            out.append(df)

        return (
            pd.concat(out, ignore_index=True)
            if out
            else pd.DataFrame(columns=[group, "node", "hub_score", "authority_score", "hub_rank", "authority_rank"])
        )

    # ---------- 3) Degree metrics ----------
    def build_degrees(
        self,
        df_edges: pd.DataFrame,
        *,
        group: str = "merged_subnetwork_id",
        src: str = "src_txn_id",
        dst: str = "dst_txn_id",
    ) -> pd.DataFrame:
        """
        Compute in/out degree per node and dense ranks per subnetwork.
        Returns: [group, node, in_txs_degree, out_txs_degree, rank_inDeg, rank_outDeg]
        """
        if df_edges.empty:
            return pd.DataFrame(columns=[group, "node", "in_txs_degree", "out_txs_degree", "rank_inDeg", "rank_outDeg"])

        out: List[pd.DataFrame] = []
        for gid, ed in df_edges.groupby(group, sort=False):
            G = self._digraph(ed, src, dst)
            if len(G) == 0:
                continue
            indeg = dict(G.in_degree())
            outdeg = dict(G.out_degree())
            nodes = list(G.nodes())
            df = pd.DataFrame(
                {
                    group: gid,
                    "node": nodes,
                    "in_txs_degree": [indeg.get(n, 0) for n in nodes],
                    "out_txs_degree": [outdeg.get(n, 0) for n in nodes],
                }
            )
            self._rank_in_group(df, group, "in_txs_degree", "rank_inDeg", ascending=False)
            # Ascending rank for out-degree to prioritise sinks (lower fan-out).
            self._rank_in_group(df, group, "out_txs_degree", "rank_outDeg", ascending=True)
            out.append(df)

        return (
            pd.concat(out, ignore_index=True)
            if out
            else pd.DataFrame(columns=[group, "node", "in_txs_degree", "out_txs_degree", "rank_inDeg", "rank_outDeg"])
        )

    # ---------- 4) Centralities ----------
    def build_centralities(
        self,
        df_edges: pd.DataFrame,
        *,
        group: str = "merged_subnetwork_id",
        src: str = "src_txn_id",
        dst: str = "dst_txn_id",
        betweenness_cutoff_n: int = 5000,
        katz_alpha: float = 0.1,
    ) -> pd.DataFrame:
        """
        Compute centrality measures and ranks per subnetwork: betweenness, harmonic,
        undirected eigenvector, directed Katz, and coreness.
        Returns centrality columns plus their within-group dense ranks.
        """
        if df_edges.empty:
            return pd.DataFrame(
                columns=[
                    group,
                    "node",
                    "betweenness",
                    "harmonic",
                    "eigenvector",
                    "katz",
                    "coreness",
                    "rank_betw",
                    "rank_harm",
                    "rank_eig",
                    "rank_katz",
                    "rank_coreness",
                ]
            )

        out: List[pd.DataFrame] = []
        for gid, ed in df_edges.groupby(group, sort=False):
            G = self._digraph(ed, src, dst)
            if len(G) == 0:
                continue
            Gu = G.to_undirected()
            N = G.number_of_nodes()

            btw = nx.betweenness_centrality(G, normalized=True) if N <= betweenness_cutoff_n else defaultdict(float)
            harm = nx.harmonic_centrality(G) if N <= 10000 else defaultdict(float)

            if N < 3:
                eig = {n: 0.0 for n in Gu.nodes()}
            else:
                try:
                    eig = nx.eigenvector_centrality_numpy(Gu)
                except Exception:
                    eig = defaultdict(float)

            try:
                katz = nx.katz_centrality_numpy(G, alpha=katz_alpha, beta=1.0)
            except Exception:
                katz = defaultdict(float)

            try:
                core = nx.core_number(Gu)
            except Exception:
                core = defaultdict(int)

            nodes = list(G.nodes())
            df = pd.DataFrame(
                {
                    group: gid,
                    "node": nodes,
                    "betweenness": [btw.get(n, 0.0) for n in nodes],
                    "harmonic": [harm.get(n, 0.0) for n in nodes],
                    "eigenvector": [eig.get(n, 0.0) for n in nodes],
                    "katz": [katz.get(n, 0.0) for n in nodes],
                    "coreness": [core.get(n, 0) for n in nodes],
                }
            )

            gb = df.groupby(group)
            df["rank_betw"] = gb["betweenness"].rank(method="dense", ascending=False)
            df["rank_harm"] = gb["harmonic"].rank(method="dense", ascending=False)
            df["rank_eig"] = gb["eigenvector"].rank(method="dense", ascending=False)
            df["rank_katz"] = gb["katz"].rank(method="dense", ascending=False)
            df["rank_coreness"] = gb["coreness"].rank(method="dense", ascending=False)
            out.append(df)

        return (
            pd.concat(out, ignore_index=True)
            if out
            else pd.DataFrame(
                columns=[
                    group,
                    "node",
                    "betweenness",
                    "harmonic",
                    "eigenvector",
                    "katz",
                    "coreness",
                    "rank_betw",
                    "rank_harm",
                    "rank_eig",
                    "rank_katz",
                    "rank_coreness",
                ]
            )
        )

    # ---------- 5) Minimum hop distance ----------
    def build_min_hops(
        self,
        df_edges: pd.DataFrame,
        *,
        group: str = "merged_subnetwork_id",
        src: str = "src_txn_id",
        dst: str = "dst_txn_id",
        src_hop: str = "min_src_hop",
        dst_hop: str = "min_dst_hop",
    ) -> pd.DataFrame:
        """
        Compute the minimum observed hop distance per node across src/dst roles.
        Returns: [group, node, hop]
        """
        if df_edges.empty:
            return pd.DataFrame(columns=[group, "node", "hop"])

        left = df_edges[[group, src, src_hop]].rename(columns={src: "node", src_hop: "hop"})
        right = df_edges[[group, dst, dst_hop]].rename(columns={dst: "node", dst_hop: "hop"})
        df = pd.concat([left, right], ignore_index=True)
        return df.groupby([group, "node"], as_index=False)["hop"].min()

    # ---------- 7) Composite scoring ----------

    def build_composite_calc(self,
        df_txn_features: pd.DataFrame,
        df_pr: pd.DataFrame,
        df_nw_txn_final: pd.DataFrame,
        *,
        nw_txn_col: str,                       # REQUIRED: exact txn-id column name in df_nw_txn_final (e.g. 'txn_id')
        group_col: str = 'merged_subnetwork_id',
        feats_txid_col: str = 'txId',
        feats_val_col: str = 'in_BTC_total',
        feats_indeg_col: str = 'in_txs_degree',
        feats_outdeg_col: str = 'out_txs_degree',
        pr_node_col: str = 'node',
        pr_value_col: str = 'pagerank',
        weights: Optional[Dict[str, float]] = None,
        invert_out_deg: bool = True,
        eps: float = 1e-12
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Build percentile-based composite using df_nw_txn_final as canonical node->subnetwork mapping.

        IMPORTANT:
        - nw_txn_col is required and must be the exact column name in df_nw_txn_final that holds txn ids.
        - Option B behaviour: keep raw pagerank = NaN where missing (traceability). pagerank_pct stays NaN.
            Use pagerank_pct_filled_for_contrib (filled with 0) to compute pr_contrib so missing PR contributes 0.
        - Final column ordering places pagerank_pct_filled_for_contrib immediately after pagerank_pct,
            and out_deg_pct BEFORE out_deg_pct_inv.
        Returns: (df_out, weights_used)
        """
        # ---------- basic checks ----------
        if nw_txn_col is None:
            raise ValueError("nw_txn_col is required. Provide the exact txn-id column name from df_nw_txn_final (e.g. 'txn_id').")

        if weights is None:
            weights = {'pr': 0.60, 'val': 0.30, 'in': 0.07, 'out': 0.03}
        if abs(sum(weights.values()) - 1.0) > 1e-6:
            raise ValueError("weights must sum to 1.0")

        # required columns check (minimal)
        req_feats = [feats_txid_col, feats_val_col, feats_indeg_col, feats_outdeg_col]
        miss = [c for c in req_feats if c not in df_txn_features.columns]
        if miss:
            raise KeyError(f"df_txn_features missing required columns: {miss}")

        if pr_node_col not in df_pr.columns or pr_value_col not in df_pr.columns:
            raise KeyError(f"df_pr must contain columns: '{pr_node_col}' and '{pr_value_col}'")

        if group_col not in df_nw_txn_final.columns or nw_txn_col not in df_nw_txn_final.columns:
            raise KeyError(f"df_nw_txn_final must contain '{group_col}' and the txn column '{nw_txn_col}'")

        # ---------- canonical mapping (one row per txn) ----------
        node_group = df_nw_txn_final[[group_col, nw_txn_col]].rename(columns={nw_txn_col: 'node'}).drop_duplicates(subset=['node']).reset_index(drop=True)
        node_group['node'] = node_group['node'].astype(str)

        # ---------- attach features ----------
        feats = df_txn_features[[feats_txid_col, feats_val_col, feats_indeg_col, feats_outdeg_col]].rename(columns={
            feats_txid_col: 'node',
            feats_val_col: 'in_BTC_total',
            feats_indeg_col: 'in_txs_degree',
            feats_outdeg_col: 'out_txs_degree'
        }).copy()
        feats['node'] = feats['node'].astype(str)

        merged = node_group.merge(feats, on='node', how='left')

        # ---------- attach pagerank (do NOT fill missing pagerank) ----------
        df_pr_cp = df_pr.rename(columns={pr_node_col: 'node', pr_value_col: 'pagerank'})[['node','pagerank']].copy()
        df_pr_cp['node'] = df_pr_cp['node'].astype(str)
        merged = merged.merge(df_pr_cp, on='node', how='left')

        # flag missing pagerank for traceability
        merged['pagerank_missing'] = merged['pagerank'].isna()

        # ---------- fill defaults for feature numerics (but NOT pagerank) ----------
        merged['in_BTC_total'] = merged['in_BTC_total'].astype(float).fillna(0.0)
        merged['in_txs_degree'] = merged['in_txs_degree'].fillna(0).astype(int)
        merged['out_txs_degree'] = merged['out_txs_degree'].fillna(0).astype(int)
        # keep pagerank NaN where missing

        # ---------- percentile computations ----------
        pct = lambda s: s.rank(method='average', pct=True)

        merged['pagerank_pct'] = merged.groupby(group_col)['pagerank'].transform(pct)
        merged['in_btc_pct']   = merged.groupby(group_col)['in_BTC_total'].transform(pct)
        merged['in_deg_pct']   = merged.groupby(group_col)['in_txs_degree'].transform(pct)
        merged['out_deg_pct']  = merged.groupby(group_col)['out_txs_degree'].transform(pct)

        # compute inverted out-degree if requested (keep both cols)
        merged['out_deg_pct_inv'] = 1.0 - merged['out_deg_pct'] if invert_out_deg else merged['out_deg_pct']

        # absolute share
        merged['in_btc_share'] = merged.groupby(group_col)['in_BTC_total'].transform(lambda s: s / (s.sum() + eps))

        # ---------- contributions: use filled pagerank_pct only for contribution arithmetic ----------
        # create helper filled percentile for PR contribution (do not overwrite pagerank_pct)
        merged['pagerank_pct_filled_for_contrib'] = merged['pagerank_pct'].fillna(0.0)

        # ensure other percentiles are non-NaN for arithmetic
        merged['in_btc_pct'] = merged['in_btc_pct'].fillna(0.0)
        merged['in_deg_pct'] = merged['in_deg_pct'].fillna(0.0)
        merged['out_deg_pct'] = merged['out_deg_pct'].fillna(0.0)
        merged['out_deg_pct_inv'] = merged['out_deg_pct_inv'].fillna(0.0)

        # ---------- weights and contributions ----------
        merged['pr_weight']  = float(weights['pr'])
        merged['val_weight'] = float(weights['val'])
        merged['in_weight']  = float(weights['in'])
        merged['out_weight'] = float(weights['out'])

        merged['pr_contrib'] = merged['pr_weight']  * merged['pagerank_pct_filled_for_contrib']
        merged['val_contrib'] = merged['val_weight'] * merged['in_btc_pct']
        merged['in_deg_contrib']  = merged['in_weight']  * merged['in_deg_pct']
        merged['out_deg_contrib'] = merged['out_weight'] * merged['out_deg_pct_inv']

        # ---------- composite & ranks ----------
        merged['composite_raw'] = merged[['pr_contrib','val_contrib','in_deg_contrib','out_deg_contrib']].sum(axis=1)
        merged['composite_pct'] = merged.groupby(group_col)['composite_raw'].transform(pct)
        merged['composite_rank'] = merged.groupby(group_col)['composite_raw'].rank(method='dense', ascending=False).astype('Int64')

        # ---------- column ordering (pagerank_pct_filled adjacent to pagerank_pct; out_deg_pct before out_deg_pct_inv) ----------
        raw_fields = ['in_BTC_total','in_txs_degree','out_txs_degree']
        pct_fields = ['pagerank_pct','pagerank_pct_filled_for_contrib','in_btc_pct','in_deg_pct','out_deg_pct','out_deg_pct_inv']
        weight_cols = ['pr_weight','val_weight','in_weight','out_weight']
        contrib_cols = ['pr_contrib','val_contrib','in_deg_contrib','out_deg_contrib']
        composite_cols = ['composite_raw','composite_pct','composite_rank']

        cols_order = []
        cols_order.extend([group_col, 'node'])
        cols_order.extend([c for c in raw_fields if c in merged.columns])

        if 'pagerank' in merged.columns:
            cols_order.append('pagerank')

        # add percentiles in the order defined above (will place pagerank_pct_filled right after pagerank_pct)
        for c in pct_fields:
            if c in merged.columns and c not in cols_order:
                cols_order.append(c)

        # add in_btc_share next
        if 'in_btc_share' in merged.columns and 'in_btc_share' not in cols_order:
            cols_order.append('in_btc_share')

        # weights, contributions, composite
        cols_order.extend([c for c in weight_cols if c in merged.columns])
        cols_order.extend([c for c in contrib_cols if c in merged.columns])
        cols_order.extend([c for c in composite_cols if c in merged.columns])

        # extras to help auditing (keep pagerank_missing as visible near the end)
        extras = ['pagerank_missing']
        for c in extras:
            if c in merged.columns and c not in cols_order:
                cols_order.append(c)

        # append any columns not explicitly ordered
        remaining = [c for c in merged.columns if c not in cols_order]
        cols_final = cols_order + remaining

        df_out = merged[cols_final].copy()
        return df_out, weights


# Correlation-based diagnostics for rank agreement across methods
# -------------------------------------------------------------------------
class ranking_comparison:
# -------------------------------------------------------------------------
    """
    Compute per-subnetwork correlation matrices across ranking methods
    and provide a simple heatmap visualisation.
    """

    def __init__(self):
        self.a = build_txn_rank()  # create instance of ClassA inside ClassB

    # Choose your rank columns (edit if you add/remove methods)
    RANK_COLS: List[str] = [
        "node_rank",                  # composite
        "pr_rank",                    # PageRank
        "hub_rank", "authority_rank", # HITS
        "rank_inDeg", "rank_outDeg",  # degrees
        "rank_betw", "rank_harm", "rank_eig", "rank_katz", "rank_coreness",  # centralities
    ]

    @staticmethod
    def make_higher_is_better(df: pd.DataFrame, rank_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """
        Convert rank columns to 'score' columns where higher is better by taking negative rank.
        Returns a copy of df and the list of created *_score columns.
        """
        out = df.copy()
        score_cols: List[str] = []
        for c in rank_cols:
            if c in out.columns:
                sc = f"{c}_score"
                out[sc] = -out[c].astype(float)
                score_cols.append(sc)
        return out, score_cols

    def corr_matrices_per_group(
        self,
        df: pd.DataFrame,
        *,
        group_col: str = "merged_subnetwork_id",
        method: str = "spearman",  # {'spearman', 'kendall', 'pearson'}
        rank_cols: Optional[List[str]] = None,
        min_nodes: int = 3
    ) -> Dict[object, pd.DataFrame]:
        """
        Build a correlation matrix (over *_score columns) for each subnetwork.
        Returns: {group_id: DataFrame}
        """
        cols = rank_cols if rank_cols is not None else [c for c in self.RANK_COLS if c in df.columns]
        df_scored, score_cols = self.make_higher_is_better(df, cols)

        # Ensure stable index/columns even for small groups
        template = pd.DataFrame(index=score_cols, columns=score_cols, dtype=float)

        corrs: Dict[object, pd.DataFrame] = {}
        for gid, g in df_scored.groupby(group_col, sort=False):
            if len(g) >= min_nodes and len(score_cols) >= 2:
                corrs[gid] = g[score_cols].corr(method=method)
            else:
                corrs[gid] = template.copy()
        return corrs

    @staticmethod
    def heatmap(df_mat: pd.DataFrame, title: str = "") -> None:
        """
        Display a matrix as a heatmap with overlaid values (rounded to 1 dp).
        """
        if df_mat is None or df_mat.empty:
            # Nothing to draw
            return

        plt.figure(figsize=(8, 6))
        im = plt.imshow(df_mat, interpolation="nearest")
        cbar = plt.colorbar(im, format="%.1f")
        cbar.ax.set_ylabel("Correlation", rotation=270, labelpad=15)

        plt.xticks(ticks=range(len(df_mat.columns)), labels=df_mat.columns, rotation=45, ha="right")
        plt.yticks(ticks=range(len(df_mat.index)), labels=df_mat.index)
        plt.title(title)

        data = df_mat.values.astype(float)
        nrows, ncols = data.shape
        vmin = np.nanmin(data) if np.isfinite(data).any() else 0.0
        vmax = np.nanmax(data) if np.isfinite(data).any() else 0.0
        mid = (vmin + vmax) / 2.0

        for i in range(nrows):
            for j in range(ncols):
                val = data[i, j]
                if np.isnan(val):
                    text = "–"
                    color = "black"
                else:
                    text = f"{val:.1f}"
                    color = "white" if val < mid else "black"
                plt.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

        plt.tight_layout()
        plt.show()

    # Optional convenience: plot a specific group's correlation heatmap directly
    def plot_group_corr(
        self,
        df: pd.DataFrame,
        gid: object,
        *,
        group_col: str = "merged_subnetwork_id",
        method: str = "spearman",
        rank_cols: Optional[List[str]] = None,
        min_nodes: int = 3,
        title_prefix: str = "Rank Correlations"
    ) -> Optional[pd.DataFrame]:
        """
        Compute and plot the correlation matrix for a single subnetwork id.
        Returns the correlation matrix (or None if unavailable).
        """
        corrs = self.corr_matrices_per_group(
            df, group_col=group_col, method=method, rank_cols=rank_cols, min_nodes=min_nodes
        )
        mat = corrs.get(gid)
        if mat is not None and not mat.empty:
            self.heatmap(mat, title=f"{title_prefix}: {gid} ({method})")
            return mat
        return None

    def average_corr_matrix(self,corr_dict, df_nw_summary, min_nodes=5,
                            group_col='merged_subnetwork_id', size_col='node_count'):
        """
        Compute elementwise mean correlation matrix across subnetworks
        with size >= min_nodes, robust to dtype/column-name mismatches.
        """
        # --- Normalise summary table ---
        summ = df_nw_summary.copy()
        # strip any accidental whitespace in column names
        summ.columns = [c.strip() for c in summ.columns]

        # resolve size column if not present
        if size_col not in summ.columns:
            alt_sizes = ['txn_count', 'tx_count', 'n_nodes', 'size', 'nodes']
            found = next((c for c in alt_sizes if c in summ.columns), None)
            if not found:
                raise KeyError(f"Size column '{size_col}' not found. Available: {list(summ.columns)}")
            size_col = found

        if group_col not in summ.columns:
            # try to find a near match (e.g., trailing space)
            candidates = [c for c in summ.columns if c.replace(" ", "") == group_col.replace(" ", "")]
            if candidates:
                group_col = candidates[0]
            else:
                raise KeyError(f"Group column '{group_col}' not found. Available: {list(summ.columns)}")

        # coerce ids and sizes
        summ[group_col] = summ[group_col].astype(str).str.strip()
        summ[size_col]  = pd.to_numeric(summ[size_col], errors='coerce')

        # valid ids by size
        valid_ids = set(summ.loc[summ[size_col] >= min_nodes, group_col])

        # --- Filter corr_dict using stringified keys ---
        # stringify keys to align with summary table
        corr_str_keys = {str(k).strip(): m for k, m in corr_dict.items() if isinstance(m, pd.DataFrame)}

        # keep only matrices for valid ids that are non-empty
        filtered_corrs = {gid: m for gid, m in corr_str_keys.items() if gid in valid_ids and not m.empty}

        if not filtered_corrs:
            # Helpful diagnostics
            n_total = len(corr_dict)
            n_valid = len(valid_ids)
            sample_valid = list(sorted(valid_ids))[:5]
            sample_corr  = list(sorted(map(lambda x: str(x).strip(), corr_dict.keys())))[:5]
            raise ValueError(
                f"No subnetworks with >= {min_nodes} nodes found in the intersection.\n"
                f"- corr_dict total keys: {n_total}\n"
                f"- df_nw_summary valid ids (>= {min_nodes}): {n_valid}\n"
                f"- sample valid ids: {sample_valid}\n"
                f"- sample corr_dict keys (stringified): {sample_corr}\n"
                "→ Check for dtype mismatches or wrong size column."
            )

        mats = list(filtered_corrs.values())

        # align axes
        common_idx = mats[0].index
        mats = [m.reindex(index=common_idx, columns=common_idx) for m in mats]

        # mean (ignore NaNs)
        avg_mat = pd.concat([m.stack() for m in mats], axis=1).mean(axis=1, skipna=True).unstack()

        return avg_mat, len(mats)

    def plot_avg_corr(self,avg_mat, n_subnetworks):
        plt.figure(figsize=(8,6))
        im = plt.imshow(avg_mat, vmin=-1, vmax=1, cmap="viridis")
        plt.colorbar(im, label="Average Spearman Correlation")
        plt.xticks(range(len(avg_mat.columns)), avg_mat.columns, rotation=45, ha="right")
        plt.yticks(range(len(avg_mat.index)), avg_mat.index)
        plt.title(f"Average Rank Correlation for {n_subnetworks} Subnetworks",
                fontsize=12, loc='center', pad=15)
        # overlay values
        for i in range(len(avg_mat.index)):
            for j in range(len(avg_mat.columns)):
                val = avg_mat.iloc[i, j]
                if np.isfinite(val):
                    plt.text(j, i, f"{val:.2f}", ha="center", va="center",
                            color="white" if abs(val) < 0.5 else "black", fontsize=8)
        plt.tight_layout()
        plt.show()


    def compare_weights_top_fraction(self,
        df_txn_features: pd.DataFrame,
        df_pr: pd.DataFrame,
        df_nw_txn_final: pd.DataFrame,
        *,
        nw_txn_col: str = "node",
        group_col: str = "merged_subnetwork_id",
        weights_list: List[Dict[str, float]],
        top_frac: float = 0.10,
        min_nodes: int = 5,
        return_details: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        For each weight configuration:
        1. Build composite using build_rank.build_composite_calc
        2. Within each subnetwork, select top `top_frac` nodes by composite_raw
        3. Compute:
            - btc_share: proportion of total inbound BTC captured by top nodes
            - pr_percentile_med: median pagerank percentile of top nodes
        4. Aggregate medians across subnetworks
        Returns:
        summary_df, details_per_weight (optional)
        """

        summaries = []
        details = {}

        # ----------------------------------------------------------
        # Iterate through all weight configurations
        # ----------------------------------------------------------
        for w in weights_list:
            wname = w.get("name", f"w_{w}")
            print(f"→ Evaluating weights: {wname}")

            # Build the composite using your existing pipeline logic
            df_comp, _weights_used = self.a.build_composite_calc(
                df_txn_features=df_txn_features,
                df_pr=df_pr,
                df_nw_txn_final=df_nw_txn_final,
                nw_txn_col=nw_txn_col,
                group_col=group_col,
                feats_txid_col="node" if "node" in df_txn_features.columns else "txId",
                feats_val_col="in_BTC_total",
                feats_indeg_col="in_txs_degree",
                feats_outdeg_col="out_txs_degree",
                pr_node_col="node",
                pr_value_col="pagerank",
                weights={k: float(v) for k, v in w.items() if k in ("pr","val","in","out")},
                invert_out_deg=True,
            )

            # ----------------------------------------------------------
            # Per-subnetwork analysis
            # ----------------------------------------------------------
            rows = []
            for sid, g in df_comp.groupby(group_col, sort=False):
                if len(g) < min_nodes:
                    continue

                g = g.copy()
                k = max(1, int(np.ceil(top_frac * len(g))))
                top = g.nlargest(k, "composite_raw")

                # --- Compute metrics ---
                btc_share = (
                    top["in_BTC_total"].sum() / g["in_BTC_total"].sum()
                    if g["in_BTC_total"].sum() > 0 else 0
                )

                if "pagerank_pct_filled_for_contrib" in g.columns:
                    pr_med = float(top["pagerank_pct_filled_for_contrib"].median() * 100)
                else:
                    g["pr_pct"] = g["pagerank"].rank(pct=True)
                    pr_med = float(top["pr_pct"].median() * 100)

                rows.append({
                    "subnet": sid,
                    "btc_share": btc_share,
                    "pr_percentile_med": pr_med,
                    "n_nodes": len(g),
                    "k_selected": k
                })

            df_detail = pd.DataFrame(rows)
            if not df_detail.empty:
                summaries.append({
                    "weight_set": wname,
                    "median_BTC_share": df_detail["btc_share"].median(),
                    "median_PR_percentile": df_detail["pr_percentile_med"].median(),
                    "n_subnetworks": df_detail["subnet"].nunique(),
                    "top_frac": top_frac
                })
            else:
                summaries.append({
                    "weight_set": wname,
                    "median_BTC_share": np.nan,
                    "median_PR_percentile": np.nan,
                    "n_subnetworks": 0,
                    "top_frac": top_frac
                })

            if return_details:
                details[wname] = df_detail

        summary_df = pd.DataFrame(summaries).sort_values("median_PR_percentile", ascending=False).reset_index(drop=True)
        return (summary_df, details) if return_details else (summary_df, {})

