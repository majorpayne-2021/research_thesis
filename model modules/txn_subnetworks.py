# -------------------------------------------------------------------------
# Module for Illicit Subnetwork Construction, Analysis, and Visualisation
# -------------------------------------------------------------------------
# This module provides all core functionality for building, deduplicating, analysing,
# and visualising illicit-only subnetworks derived from the Bitcoin transaction graph.
# It represents the subnetwork construction and exploration stage of the AML detection
# pipeline, translating transaction-level classification outputs into structured,
# interpretable subnetworks for further ranking and investigation.
#
# The module contains three main classes:
# - build_txn_subnetwork: Constructs directed illicit-to-illicit subnetworks from seed
#   transactions using breadth-first search (BFS) expansion. Includes methods for strict
#   subgraph traversal, merging overlapping subnetworks, deduplication by node subsets,
#   and filtering based on minimum network size.
# - reporting: Summarises and reports subnetwork statistics, including node and edge
#   counts, network depth, and linkage between seed and non-seed transactions. It also
#   generates summary tables for downstream analysis and audit documentation.
# - visualise_subnetwork: Generates visual representations of transaction-to-transaction
#   and address-to-address subnetworks. Provides flexible styling, node scaling, and
#   labelling options for investigative visualisation in both static and programmatic
#   formats.
#
# Together, these classes enable reproducible generation and analysis of illicit
# subnetworks, bridging the gap between classification results and investigative
# visualisation.
#
# This code forms part of the technical work developed in support of the research
# thesis titled:
# “Detection, Ranking and Visualisation of Money Laundering Networks on the Bitcoin Blockchain”
# by Jennifer Payne (RMIT University).
#
# GitHub Repository: https://github.com/majorpayne-2021/rmit_master_thesis
# Elliptic++ Dataset Source: https://github.com/git-disl/EllipticPlusPlus
# -------------------------------------------------------------------------

# Data cleaning and manipulation
import pandas as pd
import numpy as np
import math
from collections import defaultdict
from itertools import combinations
from typing import Optional

# Visualisation
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# -------------------------------------------------------------------------
class build_txn_subnetwork:
# -------------------------------------------------------------------------

    def __init__(self):
        """
        This class holds different functions to process and clean data to create a txn subnetwork.
        The default df is the dataframe assigned to the class, else, the input_df argument. 
        """
        
        self.df = pd.DataFrame()

    def strict_seed_subgraph(self,
        edges_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        seed_txn: int,
        illicit_value: str = "Illicit",
        src_col: str = "txId1",
        dst_col: str = "txId2",
        label_id_col: str = "txId",
        label_col: str = "final_class_label",
        max_hops: int | None = None,
    ):
        """
        Build a strict Illicit→Illicit forward subnetwork from a single seed (integer).

        Strict = only traverse edges whose BOTH endpoints are labelled `illicit_value`.
        This function expands the network lazily (node-by-node): at each BFS layer we
        only look up rows whose source is in the current frontier, instead of filtering
        the whole edgelist up front.

        Parameters
        ----------
        edges_df : pd.DataFrame
            Directed transaction→transaction edgelist, using only direction (no amounts).
            Must contain columns: [src_col, dst_col].
        labels_df : pd.DataFrame
            Node label table. Must contain columns: [label_id_col, label_col].
        seed_txn : int
            Starting transaction id (assumed to be Illicit; if not, traversal simply
            won't progress because we only follow Illicit→Illicit rows).
        illicit_value : str
            Label value that counts as illicit (default "Illicit").
        src_col, dst_col : str
            Column names for source/destination transaction IDs in edges_df.
        label_id_col, label_col : str
            Column names for transaction ID and its class label in labels_df.
        max_hops : int | None
            Optional breadth limit (e.g., 1 or 2). None = no limit.

        Returns
        -------
        nodes_out : pd.DataFrame
            Columns: ['txn_id', 'hop'] where 'hop' is BFS layer (0 for seed).
        edges_out : pd.DataFrame
            Columns: ['src_txn_id', 'dst_txn_id', 'src_txn_hop', 'dst_txn_hop'],
            sorted by (src_txn_hop, dst_txn_hop, src_txn_id, dst_txn_id).
        """

        # ---------- 0) Make local, typed copies ----------
        # We only need the ID columns; keep it minimal and avoid mutating caller data.
        edges = edges_df[[src_col, dst_col]].copy()
        labels = labels_df[[label_id_col, label_col]].copy()

        # Transaction IDs should be integers. Coerce to pandas nullable Int64 so
        # comparisons with an int seed are reliable (non-numeric -> <NA> and will drop out).
        edges[src_col] = pd.to_numeric(edges[src_col], errors="coerce").astype("Int64")
        edges[dst_col] = pd.to_numeric(edges[dst_col], errors="coerce").astype("Int64")
        labels[label_id_col] = pd.to_numeric(labels[label_id_col], errors="coerce").astype("Int64")

        seed_txn = int(seed_txn)  # ensure plain Python int for set/dict keys

        # ---------- 1) Fast node→label lookup ----------
        # Convert labels_df into a Series so we can map() node IDs to their class quickly.
        label_s = labels.set_index(label_id_col)[label_col]

        # ---------- 2) Build an index on source for lazy per-hop expansion ----------
        # With a (possibly non-unique) index on src, .loc[{frontier_ids}] pulls ONLY
        # the rows we need this hop (no full DataFrame scans).
        idx = edges.set_index(src_col, drop=False).sort_index()

        # ---------- 3) Initialize BFS state ----------
        visited = {seed_txn}         # all nodes included so far
        hop = {seed_txn: 0}          # BFS layer for each discovered node
        edges_seen = set()           # set of (u, v) we traversed; set dedupes repeats
        frontier = {seed_txn}        # nodes to expand at the current layer
        current_hops = 0             # how many layers we've expanded

        # ---------- 4) BFS expansion (lazy) ----------
        # Each loop expands one "ring" (hop) outward from the frontier.
        while frontier and (max_hops is None or current_hops < max_hops):

            # 4a) Limit to sources that actually exist in the index (fast; no scan).
            existing = idx.index.intersection(list(frontier))
            if len(existing) == 0:
                # None of the frontier nodes have outgoing edges in edges_df.
                break

            # 4b) Pull ONLY the rows for those sources.
            rows = idx.loc[existing, [src_col, dst_col]].copy()

            # 4c) Attach labels on this small slice and enforce STRICT Illicit→Illicit.
            #     If either endpoint is not Illicit (or label missing), we do NOT traverse it.
            rows["src_label"] = rows[src_col].map(label_s)
            rows["dst_label"] = rows[dst_col].map(label_s)
            rows = rows[rows["src_label"].eq(illicit_value) & rows["dst_label"].eq(illicit_value)]

            if rows.empty:
                # No traversable Illicit→Illicit edges from this frontier.
                break

            # 4d) Build the next frontier by following allowed edges once.
            next_frontier = set()
            for u, v in rows[[src_col, dst_col]].itertuples(index=False):
                # These come out as pandas Int64; cast to plain int for sets/dicts.
                u_i = int(u)
                v_i = int(v)

                # Record the traversed edge (dedup automatically via set)
                edges_seen.add((u_i, v_i))

                # If we haven't seen v before, discover it and assign its BFS hop.
                if v_i not in visited:
                    visited.add(v_i)
                    hop[v_i] = hop[u_i] + 1
                    next_frontier.add(v_i)

            # Move outward one ring and continue
            frontier = next_frontier
            current_hops += 1

        # ---------- 5) Assemble outputs ----------
        # 5a) Nodes table with hop (sorted by hop then ID for readability)
        nodes_out = pd.DataFrame({"txn_id": list(visited)}).astype({"txn_id": "Int64"})
        nodes_out["hop"] = nodes_out["txn_id"].map(hop)
        nodes_out = nodes_out.sort_values(["hop", "txn_id"]).reset_index(drop=True)

        # 5b) Edges table, plus hop columns for source and destination nodes
        edges_out = pd.DataFrame(sorted(edges_seen), columns=["src_txn_id", "dst_txn_id"]).astype(
            {"src_txn_id": "Int64", "dst_txn_id": "Int64"}
        )

        # Build a simple lookup: txn_id -> hop (int). Then map onto the edges.
        hop_map = dict(zip(nodes_out["txn_id"].tolist(), nodes_out["hop"].tolist()))
        edges_out["src_txn_hop"] = edges_out["src_txn_id"].map(hop_map).astype("Int64")
        edges_out["dst_txn_hop"] = edges_out["dst_txn_id"].map(hop_map).astype("Int64")

        # Sort edges primarily by BFS layer (source hop, then destination hop),
        # then by IDs for stable, readable output ordering.
        edges_out = edges_out.sort_values(
            ["src_txn_hop", "dst_txn_hop", "src_txn_id", "dst_txn_id"]
        ).reset_index(drop=True)

        return nodes_out, edges_out


    def build_subnetworks_naive(self,
        edges_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        seed_txns: list[int],
        illicit_value: str = "Illicit",
        src_col: str = "txId1",
        dst_col: str = "txId2",
        label_id_col: str = "txId",
        label_col: str = "final_class_label",
        max_hops: int | None = None,
        # ---- progress controls ----
        progress: bool = True,
        progress_every: int = 100,        # print a summary after every N seeds
        return_summary: bool = False,     # optionally return a per-seed summary DataFrame
    ):
        """
        Build a STRICT (Illicit→Illicit) forward subnetwork for EACH seed in `seed_txns`.
        Overlaps are allowed; each seed produces its own subnetwork tagged by `subnetwork_id`.

        Progress behavior:
        - Prints a kickoff line once at start.
        - Prints a compact batch summary ONLY after every `progress_every` seeds (default 100).
        - Buckets reported: 'single node network' (size==1) and 'multi-node network' (size>=2).
        - Prints ONE final overall summary with percentages.
        """
        nodes_parts, edges_parts = [], []

        total = len(seed_txns)
        processed = 0

        # Cumulative counts
        cum_single = 0
        cum_multi  = 0

        # Batch counts (reset every `progress_every`)
        batch_single = 0
        batch_multi  = 0

        # Track the largest subnetwork seen (by node count)
        max_nodes = 0
        max_nodes_seed = None

        # Optional per-seed summary capture
        summary_rows = [] if return_summary else None

        # --- kickoff line before any progress updates ---
        if progress:
            print(f"Commencing subnetwork development from seed nodes. seeds={total}  update_per_batch={progress_every}")

        for subnetwork_id, seed in enumerate(seed_txns):
            seed = int(seed)
            processed += 1

            # Build one subnetwork using your base function
            nodes_sub, edges_sub = self.strict_seed_subgraph(
                edges_df=edges_df,
                labels_df=labels_df,
                seed_txn=seed,
                illicit_value=illicit_value,
                src_col=src_col,
                dst_col=dst_col,
                label_id_col=label_id_col,
                label_col=label_col,
                max_hops=max_hops,
            )

            n_nodes = int(len(nodes_sub))
            n_edges = int(len(edges_sub))
            max_hop = int(nodes_sub["hop"].max()) if n_nodes else 0

            # Classify into the two buckets
            if n_nodes <= 1:
                cum_single += 1
                batch_single += 1
            else:
                cum_multi += 1
                batch_multi += 1

            # Track the largest
            if n_nodes > max_nodes:
                max_nodes = n_nodes
                max_nodes_seed = seed

            # Tag with subnetwork metadata and collect
            if n_nodes > 0:
                ns = nodes_sub.copy()
                ns["subnetwork_id"] = subnetwork_id
                ns["seed_txn"] = seed
                nodes_parts.append(ns)

                if n_edges > 0:
                    es = edges_sub.copy()
                    es["subnetwork_id"] = subnetwork_id
                    es["seed_txn"] = seed
                    edges_parts.append(es)

            # Optional per-seed table (no printing)
            if return_summary:
                summary_rows.append({
                    "subnetwork_id": subnetwork_id,
                    "seed_txn": seed,
                    "nodes": n_nodes,
                    "edges": n_edges,
                    "max_hop": max_hop,
                    "bucket": "single node network" if n_nodes <= 1 else "multi-node network",
                })

            # ---- print batch summary every N seeds ----
            if progress and (processed % max(1, progress_every) == 0):
                print(
                    f"[{processed:>6}/{total}] "
                    f"single node network: {batch_single}  |  multi-node network: {batch_multi}  "
                    f"(cumulative single: {cum_single}, multi: {cum_multi})  "
                    f"largest_nodes={max_nodes} (seed={max_nodes_seed})"
                )
                # reset batch counters
                batch_single = 0
                batch_multi  = 0

        # Final overall summary (one line with percentages)
        if progress:
            single_pct = (cum_single / total * 100.0) if total else 0.0
            multi_pct  = (cum_multi  / total * 100.0) if total else 0.0
            print(
                "Seed list scanning is now complete. "
                f"seeds={total}  "
                f"single node network={cum_single} ({single_pct:.1f}%)  "
                f"multi-node network={cum_multi} ({multi_pct:.1f}%)  "
                f"largest_nodes={max_nodes} (seed={max_nodes_seed})"
            )

        # Concatenate outputs (empty-safe)
        nodes_all = (pd.concat(nodes_parts, ignore_index=True)
                    if nodes_parts else pd.DataFrame(columns=["txn_id","hop","subnetwork_id","seed_txn"]))
        edges_all = (pd.concat(edges_parts, ignore_index=True)
                    if edges_parts else pd.DataFrame(columns=[
                        "src_txn_id","dst_txn_id","src_txn_hop","dst_txn_hop","subnetwork_id","seed_txn"
                    ]))

        # Tidy dtypes
        for col in ("txn_id","hop","subnetwork_id","seed_txn"):
            if col in nodes_all:
                nodes_all[col] = pd.to_numeric(nodes_all[col], errors="coerce").astype("Int64")
        for col in ("src_txn_id","dst_txn_id","src_txn_hop","dst_txn_hop","subnetwork_id","seed_txn"):
            if col in edges_all:
                edges_all[col] = pd.to_numeric(edges_all[col], errors="coerce").astype("Int64")

        if return_summary:
            seed_summary = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame(
                columns=["subnetwork_id","seed_txn","nodes","edges","max_hop","bucket"]
            )
            return nodes_all, edges_all, seed_summary

        return nodes_all, edges_all


    def deduplicate_subnetworks_by_node_subset(self,
        nodes_all: pd.DataFrame,
        edges_all: pd.DataFrame,
        id_col: str = "subnetwork_id",   # which column identifies a subnetwork
        node_col: str = "txn_id",        # node column in nodes_all
        seed_col: str = "seed_txn",      # optional: used only for progress messages
        relabel: bool = False,           # if True, remap kept subnetworks to 0..K-1
        progress: bool = True,
    ):
        """
        Remove subnetworks that are exact subsets (by node set) of another subnetwork.

        Rule:
        If nodes(S_j) ⊆ nodes(S_i) for j != i, then S_j is redundant and is removed.
        (Superset includes equality, so identical node sets keep the first and drop the rest.)

        Inputs
        ------
        nodes_all : DataFrame with at least [id_col, node_col] (e.g., ['subnetwork_id','txn_id'])
        edges_all : DataFrame with at least [id_col] (e.g., 'subnetwork_id') to filter edges
        id_col    : name of the subnetwork id column
        node_col  : name of the node id column
        seed_col  : (optional) name of the seed id column — used only for human-friendly logs
        relabel   : if True, reindex kept subnetworks to 0..K-1. If False, keep original ids.
        progress  : print a summary and which subnetworks were dropped/kept

        Returns
        -------
        nodes_keep : DataFrame: nodes for kept subnetworks (relabelled if relabel=True)
        edges_keep : DataFrame: edges for kept subnetworks (relabelled if relabel=True)
        report     : DataFrame: one row per original subnetwork with size and decision
                    columns: [id_col, 'size', 'status', 'kept_as', seed_col?]
                    - status in {'kept','dropped_subset'}
                    - kept_as = the winner id that subsumed it (NaN for kept)
        mapping    : dict old_id -> new_id (empty if relabel=False)
        """
        if nodes_all.empty:
            if progress: print("Dedup: nodes_all is empty; nothing to do.")
            return nodes_all.copy(), edges_all.copy(), pd.DataFrame(), {}

        # --- 1) Build node sets per subnetwork ------------------------------------
        # nodes_by_id[id] = set of txn_ids in that subnetwork
        grp = nodes_all.groupby(id_col)[node_col].apply(lambda s: set(pd.to_numeric(s, errors="coerce").dropna().astype(int)))
        nodes_by_id = grp.to_dict()

        # Optional: seed lookup for nice logs
        seed_of = {}
        if seed_col in nodes_all.columns:
            seed_of = (nodes_all[[id_col, seed_col]].drop_duplicates(subset=[id_col])
                    .set_index(id_col)[seed_col].to_dict())

        # Prepare list of (id, nodeset, size) and sort by size desc, then id asc (deterministic)
        items = [(int(i), s, len(s)) for i, s in nodes_by_id.items()]
        items.sort(key=lambda x: (-x[2], x[0]))  # larger first; ties broken by smaller id first

        # --- 2) Greedy keep/drop via superset check --------------------------------
        kept_ids = []         # list of kept subnetwork ids in decision order
        kept_sets = []        # corresponding node sets (same order)
        dropped = {}          # id -> winner_id that subsumed it

        for sid, sset, ssize in items:
            # If ANY already-kept set is a superset of this set, drop this one
            is_subset = False
            for kid, kset in zip(kept_ids, kept_sets):
                if kset.issuperset(sset):     # also True when equal sets
                    dropped[sid] = kid
                    is_subset = True
                    break
            if not is_subset:
                kept_ids.append(sid)
                kept_sets.append(sset)

        kept_ids_set = set(kept_ids)

        # --- 3) Filter nodes/edges to kept subnetworks -----------------------------
        nodes_keep = nodes_all[nodes_all[id_col].isin(kept_ids_set)].copy()
        edges_keep = edges_all[edges_all[id_col].isin(kept_ids_set)].copy()

        # --- 4) Optional relabel 0..K-1 for the kept ones --------------------------
        mapping = {}
        if relabel:
            mapping = {old: new for new, old in enumerate(sorted(kept_ids))}
            nodes_keep[id_col] = nodes_keep[id_col].map(mapping).astype("Int64")
            edges_keep[id_col] = edges_keep[id_col].map(mapping).astype("Int64")

        # --- 5) Build a human-friendly report --------------------------------------
        rows = []
        for sid, sset, ssize in items:
            status = "kept" if sid in kept_ids_set else "dropped_subset"
            kept_as = (sid if status == "kept" else dropped.get(sid))
            row = {id_col: sid, "size": ssize, "status": status, "kept_as": kept_as}
            if seed_of:
                row[seed_col] = seed_of.get(sid)
            rows.append(row)
        report = pd.DataFrame(rows).sort_values([ "status", id_col]).reset_index(drop=True)

        # --- 6) Progress output -----------------------------------------------------
        if progress:
            n_before = len(items)
            n_after = len(kept_ids)
            print(f"\nDedup by node-subset: {n_before} → {n_after} subnetworks kept "
                f"({n_before - n_after} removed).")
            if dropped:
                for sid, winner in sorted(dropped.items()):
                    if seed_of:
                        print(f"  - drop {sid} (seed={seed_of.get(sid)}) "
                            f"⊆ kept {winner} (seed={seed_of.get(winner)})")
                    else:
                        print(f"  - drop {sid} ⊆ kept {winner}")
            print()
            
        return nodes_keep, edges_keep, report, mapping


    def merge_subnetworks_by_node_overlap(self,
        nodes_all: pd.DataFrame,
        edges_all: pd.DataFrame,
        id_col: str = "subnetwork_id",
        seed_col: str = "seed_txn",
        node_col: str = "txn_id",
        min_shared_nodes: int = 1,
        collapse: bool = True,
        progress: bool = True,
        print_unmerged: bool = False,   # NEW: print only merged groups by default
    ):
        """
        Merge subnetworks if their node sets overlap by at least `min_shared_nodes`.
        Returns retagged views (no row loss) and, if collapse=True, deduped summaries.
        The returned DataFrames are pre-sorted for convenient display.
        """

        if nodes_all.empty:
            if progress:
                print("Merge-overlap: nodes_all is empty; nothing to merge.")
            out_nodes = nodes_all.copy()
            out_edges = edges_all.copy()
            if "merged_subnetwork_id" not in out_nodes.columns:
                out_nodes["merged_subnetwork_id"] = pd.Series(dtype="Int64")
            if "merged_subnetwork_id" not in out_edges.columns:
                out_edges["merged_subnetwork_id"] = pd.Series(dtype="Int64")
            if collapse:
                return (out_nodes, out_edges, pd.DataFrame(), pd.DataFrame())
            return (out_nodes, out_edges)

        # --- Build maps ---
        node_to_subnets = (
            nodes_all.groupby(node_col)[id_col]
            .apply(lambda s: set(pd.to_numeric(s, errors="coerce").dropna().astype(int)))
            .to_dict()
        )
        seed_of = {}
        if seed_col in nodes_all.columns:
            seed_of = (nodes_all[[id_col, seed_col]]
                    .drop_duplicates(subset=[id_col])
                    .set_index(id_col)[seed_col].to_dict())
        sub_ids = sorted(set(pd.to_numeric(nodes_all[id_col], errors="coerce").dropna().astype(int)))

        # --- Union-Find by overlap ---
        parent = {}
        def find(a):
            parent.setdefault(a, a)
            if parent[a] != a:
                parent[a] = find(parent[a])
            return parent[a]
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        if min_shared_nodes == 1:
            for subs in node_to_subnets.values():
                subs = sorted(subs)
                for other in subs[1:]:
                    union(subs[0], other)
            for sid in sub_ids:
                find(sid)
        else:
            overlap_counts = defaultdict(int)
            for subs in node_to_subnets.values():
                subs = sorted(subs)
                for i, j in combinations(subs, 2):
                    overlap_counts[(i, j)] += 1
            for (i, j), c in overlap_counts.items():
                if c >= min_shared_nodes:
                    union(i, j)
            for sid in sub_ids:
                find(sid)

        merged_id_of = {sid: find(sid) for sid in sub_ids}

        # --- Retag (no row loss) ---
        nodes_retag = nodes_all.copy()
        edges_retag = edges_all.copy()
        nodes_retag["merged_subnetwork_id"] = nodes_retag[id_col].map(merged_id_of).astype("Int64")
        edges_retag["merged_subnetwork_id"] = edges_retag[id_col].map(merged_id_of).astype("Int64")

        # --- Sort retagged outputs (so you can print directly) ---
        nodes_retag = nodes_retag.sort_values(
            ["merged_subnetwork_id", id_col, "hop", node_col]
        ).reset_index(drop=True)
        edges_retag = edges_retag.sort_values(
            ["merged_subnetwork_id", id_col, "src_txn_hop", "dst_txn_hop", "src_txn_id", "dst_txn_id"]
        ).reset_index(drop=True)

        # --- Collapsed summaries (dedup) ---
        nodes_collapsed = pd.DataFrame()
        edges_collapsed = pd.DataFrame()
        if collapse:
            seeds_in_group = (
                nodes_retag.groupby("merged_subnetwork_id")[seed_col]
                .apply(lambda s: sorted(set(map(int, s))))
                .rename("seeds_in_group")
            )
            nodes_collapsed = (
                nodes_retag.groupby(["merged_subnetwork_id", node_col])["hop"]
                .min().rename("min_hop").reset_index()
                .merge(seeds_in_group, left_on="merged_subnetwork_id", right_index=True, how="left")
            ).sort_values(
                ["merged_subnetwork_id", "min_hop", node_col]
            ).reset_index(drop=True)

            edges_collapsed = (
                edges_retag.groupby(["merged_subnetwork_id", "src_txn_id", "dst_txn_id"])
                .agg(min_src_hop=("src_txn_hop", "min"),
                    min_dst_hop=("dst_txn_hop", "min"))
                .reset_index()
            ).sort_values(
                ["merged_subnetwork_id", "src_txn_id", "dst_txn_id"]
            ).reset_index(drop=True)

        # --- Progress (summary + only merged groups unless told otherwise) ---
        if progress:
            n_before = len(sub_ids)
            n_after = len(set(merged_id_of.values()))
            print(f"Merge by node-overlap (≥{min_shared_nodes}): {n_before} → {n_after} merged subnetworks.")

            # Build groups for optional printing
            groups = {}
            for k, v in merged_id_of.items():
                groups.setdefault(v, []).append(k)

            for rep in sorted(groups):
                members = sorted(groups[rep])
                if not print_unmerged and len(members) == 1:
                    continue  # skip singletons unless explicitly requested
                seeds = [seed_of[m] for m in members if m in seed_of]
                print(f"  group {rep}: subnetwork_ids={members}" +
                    (f" seeds={sorted(set(map(int, seeds)))}" if seeds else ""))

        return (nodes_retag, edges_retag, nodes_collapsed, edges_collapsed) if collapse \
            else (nodes_retag, edges_retag)

    def filter_subnetworks_by_min_size(self,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        *,
        id_col: str = "merged_subnetwork_id",   # or "subnetwork_id"
        node_col: str = "txn_id",
        min_nodes: int = 2,                     # keep groups with >= this many nodes
        min_edges: int | None = None,           # optional extra rule (e.g., >=1 edge)
        relabel: bool = False,                  # renumber kept ids to 0..K-1
        progress: bool = True,
    ):
        """
        Drop subnetworks that are too small. By default keeps only groups with >=2 nodes.
        Works with either per-seed subnetworks ('subnetwork_id') or merged groups
        ('merged_subnetwork_id'). Returns filtered nodes/edges plus a small report and id mapping.
        """
        if nodes_df.empty:
            if progress: print("Min-size filter: nodes_df is empty; nothing to do.")
            return nodes_df.copy(), edges_df.copy(), pd.DataFrame(), {}

        # --- 1) Count nodes (unique txn) per subnetwork ---
        node_counts = (nodes_df.groupby(id_col)[node_col]
                    .nunique().rename("node_count"))

        # --- 2) (Optional) count edges per subnetwork ---
        if id_col in edges_df.columns and not edges_df.empty:
            edge_counts = edges_df.groupby(id_col).size().rename("edge_count")
        else:
            edge_counts = pd.Series(0, index=node_counts.index, name="edge_count")

        # --- 3) Decide which ids to keep ---
        keep_mask = node_counts >= min_nodes
        if min_edges is not None:
            keep_mask &= edge_counts >= int(min_edges)

        kept_ids = set(node_counts.index[keep_mask])
        dropped_ids = sorted(set(node_counts.index) - kept_ids)

        # --- 4) Filter nodes/edges ---
        nodes_keep = nodes_df[nodes_df[id_col].isin(kept_ids)].copy()
        edges_keep = edges_df[edges_df[id_col].isin(kept_ids)].copy() if id_col in edges_df.columns else edges_df.copy()

        # --- 5) (Optional) relabel to 0..K-1 for tidier downstream work ---
        mapping: dict[int, int] = {}
        if relabel and kept_ids:
            mapping = {old: new for new, old in enumerate(sorted(kept_ids))}
            nodes_keep[id_col] = nodes_keep[id_col].map(mapping).astype("Int64")
            if id_col in edges_keep.columns:
                edges_keep[id_col] = edges_keep[id_col].map(mapping).astype("Int64")

        # --- 6) Build a small report for visibility ---
        report = (pd.DataFrame({
                    id_col: node_counts.index,
                    "node_count": node_counts.values,
                })
                .merge(edge_counts, left_on=id_col, right_index=True, how="left")
                .assign(status=lambda d: d[id_col].isin(kept_ids).map({True: "kept", False: "dropped_small"}),
                        kept_as=lambda d: pd.NA)
                .sort_values([ "status", id_col])
                .reset_index(drop=True)
        )

        # --- 7) Progress line ---
        if progress:
            print(f"Min-size filter (nodes ≥{min_nodes}" +
                (f", edges ≥{min_edges}" if min_edges is not None else "") +
                f"): {len(node_counts)} → {len(kept_ids)} subnetworks kept ({len(dropped_ids)} removed).")

        return nodes_keep, edges_keep, report, mapping
    
# -------------------------------------------------------------------------
class reporting:
# -------------------------------------------------------------------------

    def __init__(self):
        """
        This class holds different functions to report on the txn subnetworks.
        The default df is the dataframe assigned to the class, else, the input_df argument. 
        """
        
        self.df = pd.DataFrame()

    def summarise_subnetworks(self,
        nodes_df: pd.DataFrame,
        edges_df: pd.DataFrame,
        *,
        id_col: str = "subnetwork_id",    # or "merged_subnetwork_id"
        node_col: str = "txn_id",
        seed_col: str | None = None,      # auto-detect if None: prefers 'seed_txn', else 'seeds_in_group'
        hop_col: str | None = None,       # auto-detect if None: prefers 'hop', else 'min_hop'
        sort_by: str = "size"             # "id" or "size"
    ) -> pd.DataFrame:
        """
        Per-subnetwork summary with txn list, counts, depth, seeds, and seed_count.
        Works with retagged (seed_txn per row) and collapsed (seeds_in_group=list) views.
        """
        if nodes_df.empty:
            return pd.DataFrame(columns=[
                id_col, "txn_ids", "node_count", "edge_count", "depth",
                "seeds", "seed_count", "linked_txn_count"
            ])

        # --- choose hop column ---
        if hop_col is None:
            hop_col = "hop" if "hop" in nodes_df.columns else ("min_hop" if "min_hop" in nodes_df.columns else None)

        # --- choose seed column ---
        if seed_col is None:
            if "seed_txn" in nodes_df.columns:
                seed_col = "seed_txn"           # retagged view
            elif "seeds_in_group" in nodes_df.columns:
                seed_col = "seeds_in_group"     # collapsed view
            else:
                seed_col = None

        # --- txn list & node count ---
        txn_lists = (
            nodes_df.groupby(id_col)[node_col]
            .apply(lambda s: sorted(set(pd.to_numeric(s, errors="coerce").dropna().astype(int))))
        )
        node_count = txn_lists.apply(len).rename("node_count")

        # --- depth (max hop) ---
        if hop_col and hop_col in nodes_df.columns:
            depth = nodes_df.groupby(id_col)[hop_col].max().rename("depth")
        else:
            depth = pd.Series(index=txn_lists.index, dtype="Int64", name="depth")

        # --- seeds (list per subnetwork) ---
        def _flatten_to_ints(series):
            vals = []
            for x in series.dropna():
                if isinstance(x, (list, tuple, set)):
                    vals.extend(list(x))
                else:
                    vals.append(x)
            out = pd.to_numeric(pd.Series(vals), errors="coerce").dropna().astype(int).tolist() if vals else []
            return sorted(set(out))

        if seed_col and seed_col in nodes_df.columns:
            any_list = nodes_df[seed_col].apply(lambda v: isinstance(v, (list, tuple, set))).any()
            if any_list:
                seed_lists = nodes_df.groupby(id_col)[seed_col].apply(_flatten_to_ints).rename("seeds")
            else:
                seed_lists = (
                    nodes_df.groupby(id_col)[seed_col]
                    .apply(lambda s: sorted(set(pd.to_numeric(s, errors="coerce").dropna().astype(int))))
                    .rename("seeds")
                )
        else:
            seed_lists = pd.Series([[]]*len(txn_lists), index=txn_lists.index, name="seeds")

        # --- seed_count ---
        seed_count = seed_lists.apply(len).rename("seed_count").astype("Int64")

        # --- edge count ---
        if not edges_df.empty and id_col in edges_df.columns:
            edge_count = edges_df.groupby(id_col).size().rename("edge_count")
        else:
            edge_count = pd.Series(0, index=txn_lists.index, name="edge_count")

        # --- assemble ---
        summary = (
            pd.DataFrame({id_col: txn_lists.index, "txn_ids": txn_lists.values})
            .merge(node_count, left_on=id_col, right_index=True)
            .merge(edge_count, left_on=id_col, right_index=True, how="left")
            .merge(depth, left_on=id_col, right_index=True, how="left")
            .merge(seed_lists, left_on=id_col, right_index=True, how="left")
            .merge(seed_count, left_on=id_col, right_index=True, how="left")
            .fillna({"edge_count": 0})
            .astype({"edge_count": "Int64"})
        )

        # linked_txn_count = nodes minus number of seeds in this group
        summary["linked_txn_count"] = summary["node_count"] - summary["seed_count"]

        # sort
        if sort_by == "id":
            summary = summary.sort_values([id_col])
        else:  # "size"
            summary = summary.sort_values(["node_count", id_col], ascending=[False, True])

        return summary.reset_index(drop=True)

    
    def filter_edges_from_seed(self,
        edges_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        seed_txn: int,
        illicit_value: str = "Illicit",
        src_col: str = "txId1",
        dst_col: str = "txId2",
        label_id_col: str = "txId",
        label_col: str = "final_class_label",
    ) -> pd.DataFrame:
        """
        Peek at the seed's outgoing edges and flag which are illicit→illicit.
        Assumes transaction IDs are integers; coerces ID columns to Int64 internally.
        Useful for sense checking subnetworks.
        """
        # Local typed copies
        edges = edges_df[[src_col, dst_col]].copy()
        labels = labels_df[[label_id_col, label_col]].copy()

        # Coerce IDs to integers (nullable Int64)
        edges[src_col] = pd.to_numeric(edges[src_col], errors="coerce").astype("Int64")
        edges[dst_col] = pd.to_numeric(edges[dst_col], errors="coerce").astype("Int64")
        labels[label_id_col] = pd.to_numeric(labels[label_id_col], errors="coerce").astype("Int64")

        seed_txn = int(seed_txn)

        # Fast node -> label lookup
        label_s = labels.set_index(label_id_col)[label_col]

        # Seed's out-edges only + labels
        out = edges[edges[src_col] == seed_txn].copy()
        out["src_label"] = out[src_col].map(label_s)
        out["dst_label"] = out[dst_col].map(label_s)
        out["both_illicit"] = out["src_label"].eq(illicit_value) & out["dst_label"].eq(illicit_value)
        return out.reset_index(drop=True)
    
    def build_subnetwork_viz_summary(self,
        df_rank: pd.DataFrame,
        df_addrtxn: pd.DataFrame,
        df_txnaddr: pd.DataFrame,
        *,
        join_to_string: bool = False,
        sep: str = "; "
    ) -> pd.DataFrame:
        """
        Build per-txn summary for the nodes in df_rank.

        Inputs
        -------
        df_rank: must include columns like
            ['merged_subnetwork_id','node','in_BTC_total','node_rank','hop', 'node_score', ...]
        df_addrtxn: expects ['input_address','txId']
        df_txnaddr: expects ['txId','output_address']
        join_to_string: if True, joins address lists into a sep-delimited string.

        Output columns
        --------------
        subnetwork_id, txn_id, illicit_flag, seed_flag, investigation_order,
        in_BTC_total, node_score (if present), input_addresses, output_addresses,
        n_inputs, n_outputs
        """

        # ---------- Canonical column names (single source of truth) ----------
        R_NODE   = "node"
        R_SUB    = "merged_subnetwork_id"
        R_INBTC  = "in_BTC_total"
        R_RANK   = "composite_rank"    
        R_SCORE  = "composite_raw"  
        R_HOP    = "hop"

        A_IN_ADDR  = "input_address"
        A_IN_TXID  = "txId"
        A_OUT_TXID = "txId"
        A_OUT_ADDR = "output_address"

        # --- Normalize IDs to string, and keep only nodes in df_rank ---
        rank = df_rank.copy()
        rank[R_NODE] = rank[R_NODE].astype(str)
        keep_nodes = set(rank[R_NODE])

        # --- Aggregate input addresses per txn ---
        if df_addrtxn is not None and not df_addrtxn.empty:
            df_in = df_addrtxn[[A_IN_ADDR, A_IN_TXID]].copy()
            df_in[A_IN_TXID] = df_in[A_IN_TXID].astype(str)
            df_in = df_in[df_in[A_IN_TXID].isin(keep_nodes)]
            inputs_agg = (
                df_in.groupby(A_IN_TXID, as_index=False)[A_IN_ADDR]
                    .apply(lambda s: sorted(set(s.dropna())))
                    .rename(columns={A_IN_TXID: R_NODE, A_IN_ADDR: "input_addresses"})
            )
        else:
            inputs_agg = pd.DataFrame(columns=[R_NODE, "input_addresses"])

        # --- Aggregate output addresses per txn ---
        if df_txnaddr is not None and not df_txnaddr.empty:
            df_out = df_txnaddr[[A_OUT_TXID, A_OUT_ADDR]].copy()
            df_out[A_OUT_TXID] = df_out[A_OUT_TXID].astype(str)
            df_out = df_out[df_out[A_OUT_TXID].isin(keep_nodes)]
            outputs_agg = (
                df_out.groupby(A_OUT_TXID, as_index=False)[A_OUT_ADDR]
                    .apply(lambda s: sorted(set(s.dropna())))
                    .rename(columns={A_OUT_TXID: R_NODE, A_OUT_ADDR: "output_addresses"})
            )
        else:
            outputs_agg = pd.DataFrame(columns=[R_NODE, "output_addresses"])

        # --- Merge input/output aggregates ---
        addr_agg = inputs_agg.merge(outputs_agg, on=R_NODE, how="outer")

        # --- Merge with rank and tidy names ---
        summary = (
            rank.merge(addr_agg, on=R_NODE, how="left")
                .rename(columns={
                    R_SUB:  "subnetwork_id",
                    R_NODE: "txn_id",
                    R_RANK: "investigation_order"
                })
        )

        # All illicit per your convention
        summary["illicit_flag"] = True

        # --- Seed flag from hop (present in df_rank) ---
        if R_HOP in summary.columns:
            summary["seed_flag"] = summary[R_HOP].eq(0).fillna(False).astype(bool)
        else:
            summary["seed_flag"] = False

        # Ensure address list columns exist and are lists
        for col in ["input_addresses", "output_addresses"]:
            if col not in summary.columns:
                summary[col] = [[] for _ in range(len(summary))]
            else:
                summary[col] = summary[col].apply(
                    lambda v: v if isinstance(v, list) else ([] if pd.isna(v) else [v])
                )

        # Counts
        summary["n_inputs"]  = summary["input_addresses"].apply(len)
        summary["n_outputs"] = summary["output_addresses"].apply(len)

        # Optional: join lists to strings
        if join_to_string:
            summary["input_addresses"]  = summary["input_addresses"].apply(lambda lst: sep.join(lst))
            summary["output_addresses"] = summary["output_addresses"].apply(lambda lst: sep.join(lst))

        # Keep relevant columns (include node_score if present)
        keep_cols = [
            "subnetwork_id", "txn_id", "illicit_flag", "seed_flag",
            "investigation_order", R_INBTC, R_SCORE,
            "input_addresses", "output_addresses",
            "n_inputs", "n_outputs"
        ]
        summary = summary[[c for c in keep_cols if c in summary.columns]]

        # Sort by subnetwork then investigation order
        if "investigation_order" in summary.columns:
            summary = summary.sort_values(["subnetwork_id", "investigation_order"]).reset_index(drop=True)
        else:
            summary = summary.sort_values(["subnetwork_id", "txn_id"]).reset_index(drop=True)

        return summary



# -------------------------------------------------------------------------
class visualise_subnetwork:
# -------------------------------------------------------------------------

    def __init__(self):
        """
        This class holds different functions to build a graph visualisation for txn subnetworks.
        The default df is the dataframe assigned to the class, else, the input_df argument. 
        """
        
        self.df = pd.DataFrame()

    def plot_txntxn_subnetwork(
        self,
        df_network_edges: pd.DataFrame,
        df_ranked: pd.DataFrame,
        subnetwork_id,
        *,
        node_label_fields=None,
        size_by=None,
        size_min=120, size_max=900,
        max_nodes=None,
        seed=42,
        top_n_label: int | None = None,
        label_by: str | None = None,
        pad_frac_x: float = 0.08,
        pad_frac_y: float = 0.08,
        label_offset: float = 0.045,
        savepath: str | None = None,
        dpi: int = 300
    ):
        """
        Visualize a single merged_subnetwork_id as a directed NetworkX graph (spring layout).

        Uses 'composite_pct' for node size and 'composite_rank' for centre labels.
        Falls back to pagerank/pr_rank if not found.
        Automatically coerces ID data types to prevent mismatch between dataframes.
        """
        EDGE_SUB = "merged_subnetwork_id"
        EDGE_SRC = "src_txn_id"
        EDGE_DST = "dst_txn_id"
        HOP_SRC  = "min_src_hop"
        HOP_DST  = "min_dst_hop"

        R_NODE   = "node"
        R_SUB    = "merged_subnetwork_id"

        # Preferred metrics
        R_SIZE_PRIMARY = "composite_pct"
        R_RANK_PRIMARY = "composite_rank"
        # Fallback metrics
        R_SIZE_FALLBK  = "pagerank"
        R_RANK_FALLBK  = "pr_rank"

        # Other optional label fields
        R_INBTC = "in_BTC_total"
        R_PR    = "pagerank"
        R_SCORE = "node_score"

        # ------------------------------------------------------------------
        # --- 1) SAFETY: Coerce datatypes to avoid mismatched joins ---
        # ------------------------------------------------------------------
        df_network_edges = df_network_edges.copy()
        df_ranked = df_ranked.copy()

        # Coerce subnetwork IDs to int64 (ignore errors)
        for df_name, df in [("df_network_edges", df_network_edges), ("df_ranked", df_ranked)]:
            if df[EDGE_SUB].dtype != "int64":
                try:
                    df[EDGE_SUB] = df[EDGE_SUB].astype("int64")
                    print(f"ℹ️ Converted {df_name}['{EDGE_SUB}'] to int64")
                except Exception:
                    print(f"⚠️ Could not convert {df_name}['{EDGE_SUB}'] to int64; leaving as {df[EDGE_SUB].dtype}")

        # Coerce node IDs to string
        if R_NODE in df_ranked.columns:
            if df_ranked[R_NODE].dtype != "object":
                df_ranked[R_NODE] = df_ranked[R_NODE].astype(str)
                print("ℹ️ Converted df_ranked['node'] to string")
        df_network_edges[EDGE_SRC] = df_network_edges[EDGE_SRC].astype(str)
        df_network_edges[EDGE_DST] = df_network_edges[EDGE_DST].astype(str)

        # ------------------------------------------------------------------
        # --- 2) Slice dataframes ---
        # ------------------------------------------------------------------
        edges = df_network_edges[df_network_edges[EDGE_SUB] == subnetwork_id].copy()
        ranks = df_ranked[df_ranked[R_SUB] == subnetwork_id].copy()

        # Handle typo
        if "min_scr_hop" in edges.columns and HOP_SRC not in edges.columns:
            edges.rename(columns={"min_scr_hop": HOP_SRC}, inplace=True)

        # ------------------------------------------------------------------
        # --- 3) Detect seeds ---
        # ------------------------------------------------------------------
        seed_nodes = set()
        if HOP_SRC in edges.columns:
            seed_nodes.update(edges.loc[edges[HOP_SRC] == 0, EDGE_SRC].astype(str))
        if HOP_DST in edges.columns:
            seed_nodes.update(edges.loc[edges[HOP_DST] == 0, EDGE_DST].astype(str))

        # ------------------------------------------------------------------
        # --- 4) Choose size and label metrics (with fallbacks) ---
        # ------------------------------------------------------------------
        missing = []
        if size_by is None:
            if R_SIZE_PRIMARY in ranks.columns:
                size_by = R_SIZE_PRIMARY
            elif R_SIZE_FALLBK in ranks.columns:
                size_by = R_SIZE_FALLBK
                missing.append(R_SIZE_PRIMARY)
            else:
                size_by = R_SCORE if R_SCORE in ranks.columns else None
                missing.append(R_SIZE_PRIMARY)
                missing.append(R_SIZE_FALLBK)

        if label_by is None:
            if R_RANK_PRIMARY in ranks.columns:
                label_by = R_RANK_PRIMARY
            elif R_RANK_FALLBK in ranks.columns:
                label_by = R_RANK_FALLBK
                missing.append(R_RANK_PRIMARY)
            elif R_PR in ranks.columns:
                # Compute pr_rank if pagerank exists
                ranks = ranks.copy()
                ranks[R_RANK_FALLBK] = ranks[R_PR].rank(ascending=False, method="dense").astype(int)
                label_by = R_RANK_FALLBK
                missing.append(R_RANK_PRIMARY)
                missing.append(R_RANK_FALLBK + " (computed from pagerank)")
            else:
                label_by = None
                missing.append(R_RANK_PRIMARY)
                missing.append(R_RANK_FALLBK)

        if missing:
            print(f"⚠️ Missing metrics {missing}. Using size_by='{size_by}', label_by='{label_by}'.")

        if node_label_fields is None:
            node_label_fields = (R_INBTC, R_PR)

        # ------------------------------------------------------------------
        # --- 5) Build graph and attach attributes ---
        # ------------------------------------------------------------------
        G = nx.DiGraph()
        G.add_edges_from(edges[[EDGE_SRC, EDGE_DST]].itertuples(index=False, name=None))

        wanted_cols = [c for c in {label_by, size_by, R_INBTC, R_PR, R_SCORE} if c and c in ranks.columns]
        if wanted_cols and R_NODE in ranks.columns:
            attr_df = ranks.set_index(R_NODE)[wanted_cols]
            for col in wanted_cols:
                nx.set_node_attributes(G, attr_df[col].to_dict(), col)

        # ------------------------------------------------------------------
        # --- 6) Compute node sizes ---
        # ------------------------------------------------------------------
        node_list = list(G.nodes())
        if not node_list:
            print(f"⚠️ No nodes found for subnetwork {subnetwork_id}.")
            return G

        raw = np.array([G.nodes[n].get(size_by, np.nan) for n in node_list], dtype=float)
        raw = np.nan_to_num(raw, nan=0.0)
        if raw.max() > raw.min():
            sizes_arr = size_min + (raw - raw.min()) / (raw.max() - raw.min()) * (size_max - size_min)
        else:
            sizes_arr = np.full_like(raw, (size_min + size_max) / 2.0)
        size_map = dict(zip(node_list, sizes_arr))

        # ------------------------------------------------------------------
        # --- 7) Labels (top N) ---
        # ------------------------------------------------------------------
        def _fmt_num(v, dp=3):
            if v is None: return "NA"
            try: return f"{float(v):.{dp}f}"
            except: return str(v)

        if top_n_label is not None:
            metric = label_by
            label_vals = pd.Series({n: G.nodes[n].get(metric, np.nan) for n in node_list})
            if metric and "rank" in metric:
                label_vals = label_vals.replace({np.nan: np.inf})
                chosen = list(label_vals.sort_values(ascending=True).head(top_n_label).index)
            else:
                label_vals = label_vals.fillna(0.0)
                chosen = list(label_vals.sort_values(ascending=False).head(top_n_label).index)
            nodes_to_label = set(chosen)
        else:
            nodes_to_label = set(node_list)

        labels = {}
        for n in node_list:
            if n not in nodes_to_label:
                continue
            parts = [f"TxnID: {n}"]
            for field in node_label_fields:
                val = G.nodes[n].get(field)
                parts.append(("PR: " + _fmt_num(val, 4)) if field == R_PR else f"{field}: {_fmt_num(val, 3)}")
            labels[n] = "\n".join(parts)

        # ------------------------------------------------------------------
        # --- 8) Draw network ---
        # ------------------------------------------------------------------
        pos = nx.spring_layout(G, k=0.7, seed=seed)
        fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)

        node_radius_pts = {n: 0.5 * float(np.sqrt(size_map[n])) for n in node_list}
        PAD = 3.0
        for u, v in G.edges():
            nx.draw_networkx_edges(
                G, pos, ax=ax, edgelist=[(u, v)], arrows=True, arrowstyle='-|>', arrowsize=15,
                edge_color="gray", width=0.8, alpha=1.0, connectionstyle="arc3,rad=0.0",
                min_source_margin=node_radius_pts.get(u, 6.0) + PAD,
                min_target_margin=node_radius_pts.get(v, 6.0) + PAD
            )

        fill_seed, edge_seed = "#cfe8ff", "#1e88e5"
        fill_normal, edge_normal = "#FFD580", "#FF8C00"
        node_colors = [fill_seed if n in seed_nodes else fill_normal for n in node_list]
        edge_colors = [edge_seed if n in seed_nodes else edge_normal for n in node_list]

        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=node_list,
            node_size=[size_map[n] for n in node_list],
            node_color=node_colors, alpha=0.95,
            linewidths=1.4, edgecolors=edge_colors
        )

        # above-node labels
        if nodes_to_label:
            label_pos = {n: (x, y + label_offset) for n, (x, y) in pos.items() if n in nodes_to_label}
            labels_subset = {n: labels[n] for n in nodes_to_label if n in labels}
            nx.draw_networkx_labels(G, label_pos, ax=ax, labels=labels_subset,
                                    font_size=7, font_color="black", verticalalignment="bottom")

        # centre labels (rank)
        rank_labels = {
            n: ("" if (G.nodes[n].get(label_by) is None or
                    (isinstance(G.nodes[n].get(label_by), float) and np.isnan(G.nodes[n].get(label_by))))
                else str(int(G.nodes[n].get(label_by))))
            for n in node_list
        }
        nx.draw_networkx_labels(G, pos, ax=ax, labels=rank_labels,
                                font_size=8, font_color="black", font_weight="bold")

        # title, limits, legend
        ax.set_title(f"Txn→Txn Directed Graph for Subnetwork {subnetwork_id}", pad=6)
        xs, ys = np.array([p[0] for p in pos.values()]), np.array([p[1] for p in pos.values()])
        dx, dy = xs.max() - xs.min(), ys.max() - ys.min()
        pad_x, pad_y = pad_frac_x * dx, pad_frac_y * dy
        ax.set_xlim(xs.min() - pad_x, xs.max() + pad_x)
        ax.set_ylim(ys.min() - pad_y, ys.max() + pad_y)
        ax.axis("off")
        legend_elements = [
            Line2D([0], [0], marker='o', linestyle='None',
                markerfacecolor=fill_seed, markeredgecolor=edge_seed,
                label='Seed Node', markersize=9),
            Line2D([0], [0], marker='o', linestyle='None',
                markerfacecolor=fill_normal, markeredgecolor=edge_normal,
                label='Non-Seed Node', markersize=9),
        ]
        ax.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=8, borderpad=0.4)

        if savepath:
            fig.savefig(savepath, dpi=dpi, bbox_inches='tight', pad_inches=0.02)
            plt.close(fig)
        else:
            plt.show()

        return G

    def plot_addraddr_subnetwork(
        self,
        df_addrtxn: pd.DataFrame,
        df_txnaddr: pd.DataFrame,
        df_rank: pd.DataFrame,
        df_network_edges: pd.DataFrame,
        merged_subnetwork_id,
        *,
        addr_size_min: float = 220,
        addr_size_max: float = 1400,
        arrowsize: float = 22,
        k: float | None = None,
        show_legend: bool = True,
        top_n_label: int | None = None,
        # ---- AUTO SIZING OPTIONS ----
        size_metric: str = "pagerank",    # "pagerank" (agg from txns) or "strength"
        size_agg: str = "sum",            # "sum" | "mean" | "max" for txn->address
        top_n_size: int | None = None,    # if set, only top-K are large
        others_size: float = 80,          # points² for non-top nodes
        uniform_size: bool = False,       # make all nodes same size (override auto sizing)
        # --- whitespace controls ---
        pad_frac_x: float = 0.08,
        pad_frac_y: float = 0.08,
        label_offset: float = 0.055,
        savepath: str | None = None,
        dpi: int = 300
    ):
        """
        Address→Address projection for one merged_subnetwork_id.

        Node size: automatically scaled by aggregated txn-level 'pagerank' (sum/mean/max per address),
                    falling back to strength if pagerank missing.
        Use uniform_size=True to make all nodes identical size.
        """
        # Canonical columns
        A_IN_TXID, A_IN_ADDR  = "txId", "input_address"
        A_OUT_TXID, A_OUT_ADDR = "txId", "output_address"
        R_NODE, R_SUB = "node", "merged_subnetwork_id"
        EDGE_SUB, EDGE_SRC, EDGE_DST = "merged_subnetwork_id", "src_txn_id", "dst_txn_id"
        HOP_SRC, HOP_DST = "min_src_hop", "min_dst_hop"

        # ---- 0) Safety: dtype coercions ----
        df_rank = df_rank.copy()
        df_network_edges = df_network_edges.copy()

        for df_name, df in [("df_rank", df_rank), ("df_network_edges", df_network_edges)]:
            if df[EDGE_SUB].dtype != "int64":
                try:
                    df[EDGE_SUB] = df[EDGE_SUB].astype("int64")
                    print(f"ℹ️ Converted {df_name}['{EDGE_SUB}'] to int64")
                except Exception:
                    print(f"⚠️ Could not convert {df_name}['{EDGE_SUB}'] to int64; leaving as {df[EDGE_SUB].dtype}")

        # ---- 1) Slice rank for this subnetwork ----
        rank_sub = df_rank.loc[df_rank[R_SUB] == merged_subnetwork_id].copy()
        if rank_sub.empty:
            print(f"No df_rank rows for subnetwork {merged_subnetwork_id}.")
            return
        rank_sub[R_NODE] = rank_sub[R_NODE].astype(str)
        illicit_txn_ids = set(rank_sub[R_NODE])

        # ---- 2) Prepare address↔txn edges ----
        ain = df_addrtxn.copy()
        aout = df_txnaddr.copy()
        ain[A_IN_TXID] = ain[A_IN_TXID].astype(str)
        aout[A_OUT_TXID] = aout[A_OUT_TXID].astype(str)
        ain = ain[ain[A_IN_TXID].isin(illicit_txn_ids)]
        aout = aout[aout[A_OUT_TXID].isin(illicit_txn_ids)]
        if ain.empty or aout.empty:
            print("No addr→txn or txn→addr edges after filtering; nothing to project.")
            return

        # ---- 3) Identify seeds ----
        seed_txns = set()
        edges_sub = df_network_edges.loc[df_network_edges[EDGE_SUB] == merged_subnetwork_id].copy()
        if not edges_sub.empty:
            edges_sub[EDGE_SRC] = edges_sub[EDGE_SRC].astype(str)
            edges_sub[EDGE_DST] = edges_sub[EDGE_DST].astype(str)
            if "min_scr_hop" in edges_sub.columns and HOP_SRC not in edges_sub.columns:
                edges_sub.rename(columns={"min_scr_hop": HOP_SRC}, inplace=True)
            if HOP_SRC in edges_sub.columns:
                seed_txns.update(edges_sub.loc[edges_sub[HOP_SRC] == 0, EDGE_SRC].tolist())
            if HOP_DST in edges_sub.columns:
                seed_txns.update(edges_sub.loc[edges_sub[HOP_DST] == 0, EDGE_DST].tolist())
            seed_txns &= illicit_txn_ids

        # ---- 4) Build addr→addr projection ----
        pairs = (
            ain.merge(aout, left_on=A_IN_TXID, right_on=A_OUT_TXID, how="inner")
            [[A_IN_ADDR, A_OUT_ADDR, A_IN_TXID]]
            .rename(columns={A_IN_TXID: "txId"})
        )
        pairs["w"] = 1.0
        wdf = (pairs.groupby([A_IN_ADDR, A_OUT_ADDR], as_index=False)["w"]
                    .sum().rename(columns={"w": "weight"}))
        if wdf.empty:
            print("No edges to plot.")
            return

        # ---- 5) Build txn→metrics maps ----
        txn_rank_map = {}
        if "composite_rank" in rank_sub.columns:
            txn_rank_map = (
                rank_sub[[R_NODE, "composite_rank"]]
                .dropna(subset=["composite_rank"])
                .set_index(R_NODE)["composite_rank"].astype(int)
                .to_dict()
            )

        # ---- 6) Address-level aggregation ----
        addr_to_txns: dict[str, set[str]] = {}
        for row in ain[[A_IN_ADDR, A_IN_TXID]].itertuples(index=False):
            addr_to_txns.setdefault(getattr(row, A_IN_ADDR), set()).add(getattr(row, A_IN_TXID))
        for row in aout[[A_OUT_ADDR, A_OUT_TXID]].itertuples(index=False):
            addr_to_txns.setdefault(getattr(row, A_OUT_ADDR), set()).add(getattr(row, A_OUT_TXID))

        addr_final_rank: dict[str, int | None] = {}
        for addr, txns in addr_to_txns.items():
            ranks = [txn_rank_map[t] for t in txns if t in txn_rank_map]
            addr_final_rank[addr] = min(ranks) if ranks else None

        seed_addrs = set(ain.loc[ain[A_IN_TXID].isin(seed_txns), A_IN_ADDR]) \
                | set(aout.loc[aout[A_OUT_TXID].isin(seed_txns), A_OUT_ADDR])

        # ---- 7) Build directed addr graph ----
        G = nx.DiGraph()
        addr_nodes = sorted(set(wdf[A_IN_ADDR]).union(set(wdf[A_OUT_ADDR])))
        G.add_nodes_from(addr_nodes)
        for r in wdf.itertuples(index=False):
            G.add_edge(getattr(r, A_IN_ADDR), getattr(r, A_OUT_ADDR), weight=float(r.weight))

        nodes_array = np.array(list(G.nodes()))

        # ---- 8) Compute node sizes (auto pagerank) ----
        size_map = {}

        if uniform_size:
            for n in nodes_array:
                size_map[n] = others_size
        else:
            def _agg(vals: list[float], how: str) -> float:
                if not vals: return 0.0
                if how == "mean": return float(np.mean(vals))
                if how == "max":  return float(np.max(vals))
                return float(np.sum(vals))

            if size_metric == "pagerank" and "pagerank" in rank_sub.columns:
                txn_pr_map = (
                    rank_sub[[R_NODE, "pagerank"]]
                    .dropna(subset=["pagerank"])
                    .set_index(R_NODE)["pagerank"].astype(float)
                    .to_dict()
                )
                addr_metric = {}
                for addr, txns in addr_to_txns.items():
                    vals = [txn_pr_map.get(t, 0.0) for t in txns if t in txn_pr_map]
                    addr_metric[addr] = _agg(vals, size_agg)
                metric_vals = np.array([addr_metric.get(n, 0.0) for n in nodes_array], dtype=float)
            else:
                print("⚠️ pagerank missing in df_rank; falling back to strength for sizing.")
                in_w  = dict(G.in_degree(weight="weight"))
                out_w = dict(G.out_degree(weight="weight"))
                metric_vals = np.array([in_w.get(n, 0.0) + out_w.get(n, 0.0) for n in nodes_array], dtype=float)

            if metric_vals.size:
                vmin, vmax = float(metric_vals.min()), float(metric_vals.max())
                scaled = (addr_size_min + (metric_vals - vmin) * (addr_size_max - addr_size_min) / (vmax - vmin)
                        if vmax > vmin else np.full_like(metric_vals, (addr_size_min + addr_size_max)/2.0))
            else:
                scaled = np.full(len(nodes_array), (addr_size_min + addr_size_max)/2.0, dtype=float)

            if top_n_size and top_n_size > 0 and top_n_size < len(nodes_array):
                order = np.argsort(-metric_vals)
                top_idx = set(order[:top_n_size].tolist())
                for i, n in enumerate(nodes_array):
                    size_map[n] = float(scaled[i]) if i in top_idx else float(others_size)
            else:
                for i, n in enumerate(nodes_array):
                    size_map[n] = float(scaled[i])

        # ---- 9) Layout + colours ----
        pos = nx.spring_layout(G, seed=42, k=k)
        fill_seed, edge_seed = "#cfe8ff", "#1e88e5"
        fill_norm, edge_norm = "#FFD580", "#FF8C00"
        node_list   = list(G.nodes())
        node_colors = [fill_seed if n in seed_addrs else fill_norm for n in node_list]
        edge_colors = [edge_seed if n in seed_addrs else edge_norm for n in node_list]

        fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)

        def _radius(area_pts2: float) -> float:
            return math.sqrt(max(area_pts2, 1.0) / math.pi)
        size_lookup = {n: size_map.get(n, addr_size_min) for n in node_list}

        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=node_list,
            node_size=[size_map[n] for n in node_list],
            node_color=node_colors, edgecolors=edge_colors, linewidths=1.4, alpha=0.95
        )
        for u, v in G.edges():
            ru = _radius(size_lookup.get(u, addr_size_min))
            rv = _radius(size_lookup.get(v, addr_size_min))
            nx.draw_networkx_edges(
                G, pos, ax=ax, edgelist=[(u, v)], arrows=True, arrowstyle='-|>', arrowsize=arrowsize,
                width=1.0, alpha=1.0, edge_color="#9e9e9e",
                connectionstyle='arc3,rad=0.06',
                min_source_margin=0.9 * ru,
                min_target_margin=1.1 * rv
            )

        # ---- 10) Labels ----
        rank_labels = {}
        for n in node_list:
            r = addr_final_rank.get(n)
            rank_labels[n] = "" if (r is None or (isinstance(r, float) and np.isnan(r))) else str(int(r))
        nx.draw_networkx_labels(G, pos, labels=rank_labels,
                                font_size=8, font_color="black", font_weight="bold", ax=ax)

        if top_n_label is None:
            nodes_to_label = set(node_list)
        else:
            metric_series = pd.Series({n: addr_final_rank.get(n, np.inf) for n in node_list})
            nodes_to_label = set(metric_series.sort_values(ascending=True).head(top_n_label).index)

        addr_labels = {n: n for n in node_list if n in nodes_to_label}
        addr_text_pos = {n: (x, y + label_offset) for n, (x, y) in pos.items() if n in addr_labels}
        nx.draw_networkx_labels(G, addr_text_pos, labels=addr_labels,
                                font_size=7, font_color="black", ax=ax)

        # ---- 11) Title, limits, legend ----
        ax.set_title(f"Addr→Addr Directed Graph for Subnetwork {merged_subnetwork_id}", pad=6)
        xs = np.array([p[0] for p in pos.values()]) if pos else np.array([0.0])
        ys = np.array([p[1] for p in pos.values()]) if pos else np.array([0.0])
        dx = float(xs.max() - xs.min()) if xs.size else 1.0
        dy = float(ys.max() - ys.min()) if ys.size else 1.0
        pad_x = pad_frac_x * dx if dx > 0 else pad_frac_x
        pad_y = pad_frac_y * dy if dy > 0 else pad_frac_y
        ax.set_xlim(xs.min() - pad_x, xs.max() + pad_x)
        ax.set_ylim(ys.min() - pad_y, ys.max() + pad_y)
        ax.margins(x=0.01, y=0.01)
        ax.axis("off")

        if show_legend:
            legend_elements = [
                Line2D([0], [0], marker='o', linestyle='None',
                    markerfacecolor=fill_seed, markeredgecolor=edge_seed,
                    label='Seed-linked address', markersize=9),
                Line2D([0], [0], marker='o', linestyle='None',
                    markerfacecolor=fill_norm, markeredgecolor=edge_norm,
                    label='Other address', markersize=9),
            ]
            ax.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=8, borderpad=0.4)

        if savepath:
            fig.savefig(savepath, dpi=dpi, bbox_inches='tight', pad_inches=0.02)
            plt.close(fig)
        else:
            plt.show()

        return G
