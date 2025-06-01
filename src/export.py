"""
Export functionality for generating CSV reports from SLTItemGen analysis results.

This module provides functions to generate structured CSV exports for final items,
analysis summaries, and removed items from the AI-GENIE pipeline.
"""

import pandas as pd
import numpy as np
import re
import streamlit as st


def get_item_text_from_label(item_label: str, confirmed_items: list[str]) -> str:
    """Maps an 'Item X' label back to the actual item text.
    
    Args:
        item_label: Label in format 'Item X' where X is a number.
        confirmed_items: List of actual item texts from st.session_state.previous_items.
        
    Returns:
        The actual item text, or the original label if mapping fails.
    """
    try:
        # Extract number from "Item X" format (handle various separators)
        match = re.search(r'\bItem[_\s-]*?(\d+)\b', item_label)
        if match:
            item_num = int(match.group(1))
            # Convert to 0-based index
            index = item_num - 1
            if 0 <= index < len(confirmed_items):
                return confirmed_items[index]
    except (ValueError, IndexError):
        pass
    
    # Return original label if mapping fails
    return item_label


def format_items_with_text(items: list[str], confirmed_items: list[str]) -> str:
    """Formats a list of 'Item X' labels with actual item text for display.
    
    Args:
        items: List of item labels in 'Item X' format.
        confirmed_items: List of actual item texts.
        
    Returns:
        Formatted string with numbered list showing actual item text.
    """
    formatted_lines = []
    for i, item_label in enumerate(items):
        actual_text = get_item_text_from_label(item_label, confirmed_items)
        formatted_lines.append(f"{i+1}. {actual_text}")
    return "\n".join(formatted_lines)


def generate_final_items_csv() -> pd.DataFrame:
    """Generates the final items CSV dataset with all relevant metadata.

    Returns:
        DataFrame with final stable items and their characteristics.
    """
    # Get final stable items and their data
    stable_items = st.session_state.get("bootega_stable_items", [])
    stability_scores = st.session_state.get("bootega_final_stability_scores", {}) or {}
    final_communities = st.session_state.get("bootega_final_community_membership", {}) or {}
    confirmed_items = st.session_state.get("previous_items", []) or []
    uva_items = st.session_state.get("uva_final_items", []) or []

    if not stable_items:
        return pd.DataFrame()  # Return empty DataFrame if no stable items

    # Get configuration info
    network_method = st.session_state.get("network_method_select", "Unknown")
    input_matrix = st.session_state.get("input_matrix_select", "Unknown")
    embedding_method = "Dense" if input_matrix == "Dense Embeddings" else "Sparse_TFIDF"

    # Build the dataset
    csv_data = []
    for i, item_label in enumerate(stable_items):
        # Get actual item text
        actual_text = get_item_text_from_label(item_label, confirmed_items)
        
        # Get community info (with null safety)
        community_id = final_communities.get(item_label, -1) if final_communities else -1
        
        # Calculate community size (with null safety)
        if final_communities and community_id != -1:
            community_size = sum(1 for comm in final_communities.values() if comm == community_id)
        else:
            community_size = 1

        # Get stability score (with null safety)
        stability = stability_scores.get(item_label, np.nan) if stability_scores else np.nan

        # Determine retention status
        uva_retained = item_label in uva_items if uva_items else True
        bootega_retained = True  # By definition, if it's in stable_items

        # Get original generation order (extract from Item X label)
        try:
            match = re.search(r'\bItem[_\s-]*?(\d+)\b', item_label)
            generation_order = int(match.group(1)) if match else i + 1
        except (ValueError, AttributeError):
            generation_order = i + 1

        csv_data.append({
            'Item_ID': i + 1,
            'Item_Text': actual_text,
            'Original_Item_Label': item_label,
            'Community_ID': community_id,
            'Community_Size': community_size,
            'Stability_Score': round(stability, 4) if not pd.isna(stability) else None,
            'UVA_Retained': uva_retained,
            'bootEGA_Retained': bootega_retained,
            'Generation_Order': generation_order,
            'Embedding_Method': embedding_method,
            'Network_Method': network_method,
            'Analysis_Timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        })

    return pd.DataFrame(csv_data)


def generate_analysis_summary_csv() -> pd.DataFrame:
    """Generates the analysis summary CSV with pipeline metrics and parameters.

    Returns:
        DataFrame with key analysis metrics and configuration parameters.
    """
    # Get metrics
    initial_nmi = st.session_state.get("bootega_initial_nmi_compared", np.nan)
    final_nmi = st.session_state.get("bootega_final_nmi", np.nan)

    # Get TEFI (need to find the correct key based on current settings)
    network_method = st.session_state.get("network_method_select", "TMFG")
    input_matrix = st.session_state.get("input_matrix_select", "Dense Embeddings")
    matrix_suffix = "dense" if input_matrix == "Dense Embeddings" else "sparse"
    method_prefix = "tmfg" if network_method == "TMFG" else "glasso"
    tefi_key = f"tefi_{method_prefix}_{matrix_suffix}"
    tefi_score = st.session_state.get(tefi_key, np.nan)

    # Get community info
    final_communities = st.session_state.get("bootega_final_community_membership", {})
    num_communities = len(set(c for c in final_communities.values() if c != -1)) if final_communities else 0

    # Count items at each stage
    initial_count = len(st.session_state.get("previous_items", []))
    uva_count = len(st.session_state.get("uva_final_items", []))
    final_count = len(st.session_state.get("bootega_stable_items", []))

    # Get parameters
    uva_threshold = st.session_state.get("uva_threshold", np.nan)
    bootega_n_bootstraps = st.session_state.get("bootega_n_bootstraps", np.nan)
    bootega_stability_threshold = st.session_state.get("bootega_stability_threshold", np.nan)

    # Calculate average stability
    stability_scores = st.session_state.get("bootega_final_stability_scores", {})
    avg_stability = np.mean(list(stability_scores.values())) if stability_scores else np.nan

    summary_data = {
        'Metric': [
            'Initial_Item_Count', 'Post_UVA_Count', 'Final_Stable_Count',
            'Items_Removed_UVA', 'Items_Removed_bootEGA',
            'TEFI_Score', 'NMI_Initial', 'NMI_Final', 'NMI_Improvement',
            'Final_Communities', 'Average_Stability_Score',
            'Network_Method', 'Embedding_Method', 'UVA_Threshold',
            'bootEGA_Bootstrap_Samples', 'bootEGA_Stability_Threshold',
            'Analysis_Timestamp', 'Focus_Area'
        ],
        'Value': [
            initial_count, uva_count, final_count,
            initial_count - uva_count, uva_count - final_count,
            round(tefi_score, 4) if not pd.isna(tefi_score) else None,
            round(initial_nmi, 4) if not pd.isna(initial_nmi) else None,
            round(final_nmi, 4) if not pd.isna(final_nmi) else None,
            round(final_nmi - initial_nmi, 4) if not pd.isna(final_nmi) and not pd.isna(initial_nmi) else None,
            num_communities,
            round(avg_stability, 4) if not pd.isna(avg_stability) else None,
            network_method,
            "Dense" if input_matrix == "Dense Embeddings" else "Sparse_TFIDF",
            round(uva_threshold, 4) if not pd.isna(uva_threshold) else None,
            int(bootega_n_bootstraps) if not pd.isna(bootega_n_bootstraps) else None,
            round(bootega_stability_threshold, 4) if not pd.isna(bootega_stability_threshold) else None,
            pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            st.session_state.get("focus_area_selectbox", "Unknown")
        ]
    }

    return pd.DataFrame(summary_data)


def generate_removed_items_csv() -> pd.DataFrame:
    """Generates the removed items CSV with details on all filtered items.

    Returns:
        DataFrame with removed items from both UVA and bootEGA stages.
    """
    removed_data = []
    confirmed_items = st.session_state.get("previous_items", [])

    # Add UVA removed items
    uva_removed = st.session_state.get("uva_removed_log", [])
    if uva_removed:
        for item_label, wto_score in uva_removed:
            actual_text = get_item_text_from_label(item_label, confirmed_items)
            removed_data.append({
                'Item_Text': actual_text,
                'Original_Item_Label': item_label,
                'Removal_Stage': 'UVA',
                'Removal_Reason': f'wTO >= {st.session_state.get("uva_threshold", 0.20):.2f}',
                'Score': round(wto_score, 4),
                'Iteration': 1,  # UVA is iterative, but we don't track which iteration
                'Timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            })

    # Add bootEGA removed items
    bootega_removed = st.session_state.get("bootega_removed_log", [])
    if bootega_removed:
        stability_threshold = st.session_state.get("bootega_stability_threshold", 0.75)
        for item_label, stability_score, iteration in bootega_removed:
            actual_text = get_item_text_from_label(item_label, confirmed_items)
            removed_data.append({
                'Item_Text': actual_text,
                'Original_Item_Label': item_label,
                'Removal_Stage': 'bootEGA',
                'Removal_Reason': f'Stability < {stability_threshold:.2f}',
                'Score': round(stability_score, 4),
                'Iteration': iteration,
                'Timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            })

    return pd.DataFrame(removed_data)
