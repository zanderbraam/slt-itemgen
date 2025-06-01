"""
PDF Report Generation for SLTItemGen Analysis Results.

This module provides comprehensive PDF report generation functionality for the AI-GENIE
pipeline, including visualizations, analysis results, and formatted data tables.
"""

import io
import tempfile
from datetime import datetime
from typing import Any
import re  # Import needed for label extraction

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    KeepTogether,
)
from reportlab.platypus.flowables import HRFlowable

from .export import get_item_text_from_label, format_items_with_text


class SLTReportTemplate(BaseDocTemplate):
    """Custom report template with headers and footers."""
    
    def __init__(self, filename: str, **kwargs):
        BaseDocTemplate.__init__(self, filename, pagesize=letter, **kwargs)
        
        # Page setup
        self.page_width = letter[0]
        self.page_height = letter[1]
        self.margin = 72  # 1 inch margins
        
        # Create frames for content
        frame = Frame(
            self.margin, self.margin, 
            self.page_width - 2 * self.margin, 
            self.page_height - 2 * self.margin,
            id='main_frame', 
            topPadding=36, bottomPadding=36
        )
        
        # Create page template
        template = PageTemplate(id='main_template', frames=[frame])
        template.beforePage = self.add_header_footer
        self.addPageTemplates([template])
        
        # Report metadata
        self.report_title = "SLTItemGen: AI-GENIE Analysis Report"
        self.generated_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        
    def add_header_footer(self, canvas, doc):
        """Add header and footer to each page."""
        canvas.saveState()
        
        # Header
        canvas.setFont('Helvetica-Bold', 10)
        canvas.setFillColor(colors.darkblue)
        canvas.drawString(self.margin, self.page_height - 50, self.report_title)
        
        # Footer
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.grey)
        footer_text = f"Generated on {self.generated_date}"
        canvas.drawString(self.margin, 30, footer_text)
        
        # Page number
        page_num = f"Page {doc.page}"
        canvas.drawRightString(self.page_width - self.margin, 30, page_num)
        
        # Header line
        canvas.setStrokeColor(colors.darkblue)
        canvas.setLineWidth(1)
        canvas.line(self.margin, self.page_height - 60, 
                   self.page_width - self.margin, self.page_height - 60)
        
        canvas.restoreState()


def create_styles() -> dict[str, ParagraphStyle]:
    """Create custom paragraph styles for the report."""
    styles = getSampleStyleSheet()
    
    custom_styles = {
        'Title': ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue,
            fontName='Helvetica-Bold'
        ),
        'Heading1': ParagraphStyle(
            'CustomHeading1',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkblue,
            fontName='Helvetica-Bold',
            borderWidth=1,
            borderColor=colors.darkblue,
            borderPadding=5,
            backColor=colors.lightblue,
        ),
        'Heading2': ParagraphStyle(
            'CustomHeading2',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=8,
            spaceBefore=16,
            textColor=colors.darkblue,
            fontName='Helvetica-Bold'
        ),
        'Normal': ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_JUSTIFY,
            fontName='Helvetica'
        ),
        'CenteredNormal': ParagraphStyle(
            'CenteredNormal',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_CENTER,
            fontName='Helvetica'
        ),
        'Code': ParagraphStyle(
            'CodeStyle',
            parent=styles['Code'],
            fontSize=8,
            fontName='Courier',
            backColor=colors.lightgrey,
            borderWidth=1,
            borderColor=colors.grey,
            borderPadding=5
        )
    }
    
    return custom_styles


def create_summary_table(data: dict[str, Any]) -> Table:
    """Create a formatted summary table."""
    table_data = [['Metric', 'Value']]
    
    for key, value in data.items():
        # Format the key (remove underscores, title case)
        formatted_key = key.replace('_', ' ').title()
        
        # Format the value
        if isinstance(value, float):
            if pd.isna(value):
                formatted_value = "N/A"
            else:
                formatted_value = f"{value:.4f}"
        elif value is None:
            formatted_value = "N/A"
        else:
            formatted_value = str(value)
            
        table_data.append([formatted_key, formatted_value])
    
    table = Table(table_data, colWidths=[3*inch, 2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    return table


def create_items_table(items: list[str], title: str = "Items") -> Table:
    """Create a formatted table for displaying items."""
    table_data = [['#', 'Item Text']]
    
    for i, item in enumerate(items, 1):
        # Wrap long text
        if len(item) > 80:
            item = item[:77] + "..."
        table_data.append([str(i), item])
    
    table = Table(table_data, colWidths=[0.5*inch, 5.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (0, -1), 'CENTER'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.beige, colors.white]),
    ]))
    
    return table


def save_plot_to_bytes(fig: plt.Figure) -> io.BytesIO:
    """Save matplotlib figure to bytes for PDF inclusion."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buf.seek(0)
    return buf


def generate_network_plot_for_pdf(
    graph: nx.Graph, 
    membership: dict[str | int, int] | None = None,
    title: str = "Network Analysis",
    figsize: tuple[float, float] = (8, 6)
) -> plt.Figure:
    """Generate a network plot specifically formatted for PDF inclusion."""
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    if graph is None or len(graph.nodes()) == 0:
        ax.text(0.5, 0.5, 'No network data available', 
                ha='center', va='center', transform=ax.transAxes, 
                fontsize=14, style='italic')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.axis('off')
        return fig
    
    pos = nx.spring_layout(graph, seed=42)
    
    if membership:
        # Create color mapping
        unique_communities = sorted(list(set(membership.values())))
        valid_communities = [c for c in unique_communities if c != -1]
        
        if valid_communities:
            cmap = plt.get_cmap('viridis', len(valid_communities))
            color_map = {comm_id: cmap(i) for i, comm_id in enumerate(valid_communities)}
        else:
            color_map = {}
        color_map[-1] = 'grey'
        
        node_colors = [color_map.get(membership.get(node, -1), 'grey') 
                      for node in graph.nodes()]
        
        # Draw network with community colors
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, 
                              node_size=300, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(graph, pos, alpha=0.5, edge_color='grey', ax=ax)
        
        # Add legend
        legend_elements = []
        if -1 in unique_communities:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            label='Isolated', markerfacecolor='grey', 
                                            markersize=8))
        for comm_id in valid_communities:
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            label=f'Community {comm_id}', 
                                            markerfacecolor=color_map[comm_id], 
                                            markersize=8))
        
        if legend_elements:
            ax.legend(handles=legend_elements, title="Communities", 
                     loc='upper right', bbox_to_anchor=(1.15, 1))
    else:
        # Simple network without communities
        nx.draw_networkx_nodes(graph, pos, node_color='skyblue', 
                              node_size=300, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(graph, pos, alpha=0.5, edge_color='grey', ax=ax)
    
    # Always show item number labels in PDF (extract number from 'Item X' format)
    labels = {}
    for node in graph.nodes():
        # Attempt to extract number if label is like 'Item X'
        match = re.search(r'\d+$', str(node))
        if match:
            labels[node] = match.group(0)
        else:
            labels[node] = str(node)  # Fallback to full node name
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10, ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    return fig


def generate_pdf_report() -> bytes:
    """Generate comprehensive PDF report from current session state.
    
    Returns:
        PDF content as bytes.
    """
    # Create temporary file for PDF generation
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    temp_filename = temp_file.name
    temp_file.close()
    
    try:
        # Create document
        doc = SLTReportTemplate(temp_filename)
        styles = create_styles()
        story = []
        
        # ====================
        # COVER PAGE
        # ====================
        story.append(Spacer(1, 2*inch))
        story.append(Paragraph("SLTItemGen Analysis Report", styles['Title']))
        story.append(Spacer(1, 0.5*inch))
        
        # Analysis metadata
        focus_area = st.session_state.get("assessment_topic", "Unknown")
        story.append(Paragraph(f"<b>Focus Area:</b> {focus_area}", styles['CenteredNormal']))
        story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", 
                              styles['CenteredNormal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Key metrics summary
        final_items_count = len(st.session_state.get("bootega_stable_items", []))
        initial_items_count = len(st.session_state.get("previous_items", []))
        
        if final_items_count > 0:
            story.append(Paragraph("<b>Analysis Summary</b>", styles['Heading2']))
            summary_data = {
                'Initial Items Generated': initial_items_count,
                'Final Stable Items': final_items_count,
                'Retention Rate': f"{(final_items_count/initial_items_count)*100:.1f}%" if initial_items_count > 0 else "N/A",
                'Analysis Pipeline': "AI-GENIE (6-Phase)"
            }
            story.append(create_summary_table(summary_data))
        
        story.append(PageBreak())
        
        # ====================
        # EXECUTIVE SUMMARY
        # ====================
        story.append(Paragraph("Executive Summary", styles['Heading1']))
        
        # Generate executive summary text
        exec_summary = f"""
        This report presents the results of an AI-GENIE analysis conducted using SLTItemGen 
        for the assessment domain of <b>{focus_area}</b>. The analysis utilized a 6-phase 
        pipeline to generate, refine, and validate psychometric items suitable for measuring 
        communicative participation outcomes in children aged 6-11 with communication difficulties.
        """
        
        if final_items_count > 0:
            exec_summary += f"""
            <br/><br/>
            <b>Key Findings:</b><br/>
            • Started with {initial_items_count} AI-generated items<br/>
            • Applied network-based filtering (EGA) and redundancy analysis (UVA)<br/>
            • Conducted bootstrap stability validation (bootEGA)<br/>
            • Resulted in {final_items_count} high-quality, stable items<br/>
            • Achieved {(final_items_count/initial_items_count)*100:.1f}% retention rate
            """
            
            # Add network analysis results if available
            network_method = st.session_state.get("network_method_select", "Unknown")
            tefi_key = f"tefi_{network_method.lower()}_dense"
            tefi_score = st.session_state.get(tefi_key, np.nan)
            
            if not pd.isna(tefi_score):
                exec_summary += f"<br/>• Network fit (TEFI): {tefi_score:.4f}"
                
        story.append(Paragraph(exec_summary, styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # ====================
        # METHODOLOGY
        # ====================
        story.append(Paragraph("Methodology", styles['Heading1']))
        
        methodology_text = """
        The AI-GENIE pipeline implements a systematic approach to psychometric item development:
        <br/><br/>
        <b>Phase 1: Item Generation</b><br/>
        Large Language Model (GPT-4o) generates initial item pool using domain-specific prompts 
        and few-shot examples.
        <br/><br/>
        <b>Phase 2: Text Embedding</b><br/>
        Items are converted to numerical representations using OpenAI embeddings for semantic 
        analysis.
        <br/><br/>
        <b>Phase 3: Exploratory Graph Analysis (EGA)</b><br/>
        Network construction using TMFG or EBICglasso methods, followed by community detection 
        to identify item clusters.
        <br/><br/>
        <b>Phase 4: Unique Variable Analysis (UVA)</b><br/>
        Iterative removal of redundant items based on weighted topological overlap (wTO) to 
        reduce multicollinearity.
        <br/><br/>
        <b>Phase 5: Bootstrap EGA (bootEGA)</b><br/>
        Stability validation through bootstrap resampling to ensure robust community structure.
        <br/><br/>
        <b>Phase 6: Final Validation</b><br/>
        Quality assessment and export of validated item set with comprehensive metadata.
        """
        
        story.append(Paragraph(methodology_text, styles['Normal']))
        story.append(PageBreak())
        
        # ====================
        # GENERATED ITEMS
        # ====================
        if st.session_state.get("previous_items"):
            story.append(Paragraph("Generated Items", styles['Heading1']))
            
            initial_items = st.session_state.get("previous_items", [])
            story.append(Paragraph(f"The following {len(initial_items)} items were generated for the domain of <b>{focus_area}</b>:", 
                                 styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
            
            # Create items table
            items_table = create_items_table(initial_items, "Generated Items")
            story.append(items_table)
            story.append(PageBreak())
        
        # ====================
        # NETWORK ANALYSIS
        # ====================
        network_method = st.session_state.get("network_method_select", "TMFG")
        input_matrix = st.session_state.get("input_matrix_select", "Dense Embeddings")
        matrix_suffix = "dense" if input_matrix == "Dense Embeddings" else "sparse"
        method_prefix = "tmfg" if network_method == "TMFG" else "glasso"
        
        graph_key = f"graph_{method_prefix}_{matrix_suffix}"
        community_key = f"community_membership_{method_prefix}_{matrix_suffix}"
        
        current_graph = st.session_state.get(graph_key)
        current_communities = st.session_state.get(community_key)
        
        story.append(Paragraph("Network Analysis Results", styles['Heading1']))
        
        if current_graph:
            story.append(Paragraph(f"<b>Method:</b> {network_method} with {input_matrix}", styles['Normal']))
            
            # Network metrics
            num_nodes = len(current_graph.nodes())
            num_edges = len(current_graph.edges())
            num_communities = len(set(current_communities.values())) if current_communities else 0
            
            network_summary = {
                'Network Method': network_method,
                'Embedding Type': input_matrix,
                'Number of Items': num_nodes,
                'Network Edges': num_edges,
                'Detected Communities': num_communities
            }
            
            # Add TEFI if available
            tefi_key = f"tefi_{method_prefix}_{matrix_suffix}"
            tefi_score = st.session_state.get(tefi_key, np.nan)
            if not pd.isna(tefi_score):
                network_summary['TEFI Score'] = f"{tefi_score:.4f}"
            
            story.append(create_summary_table(network_summary))
            story.append(Spacer(1, 0.3*inch))
            
            # Network visualization
            network_fig = generate_network_plot_for_pdf(
                current_graph, 
                current_communities,
                f"{network_method} Network Analysis"
            )
            
            plot_buf = save_plot_to_bytes(network_fig)
            plt.close(network_fig)  # Clean up
            
            network_img = Image(plot_buf, width=6*inch, height=4.5*inch)
            story.append(network_img)
            
        else:
            story.append(Paragraph("Network analysis not completed.", styles['Normal']))
            
        story.append(PageBreak())
        
        # ====================
        # UVA RESULTS
        # ====================
        if st.session_state.get("uva_status") == "Completed":
            story.append(Paragraph("Unique Variable Analysis (UVA)", styles['Heading1']))
            
            uva_threshold = st.session_state.get("uva_threshold", 0.20)
            uva_removed = st.session_state.get("uva_removed_log", [])
            uva_final = st.session_state.get("uva_final_items", [])
            
            story.append(Paragraph(f"UVA was performed with a weighted topological overlap (wTO) threshold of {uva_threshold:.2f}. "
                                 f"Items with wTO values exceeding this threshold were iteratively removed to reduce redundancy.",
                                 styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
            
            uva_summary = {
                'wTO Threshold': f"{uva_threshold:.2f}",
                'Items Removed': len(uva_removed),
                'Items Retained': len(uva_final),
                'Reduction Rate': f"{(len(uva_removed)/(len(uva_removed)+len(uva_final)))*100:.1f}%" if (len(uva_removed)+len(uva_final)) > 0 else "N/A"
            }
            story.append(create_summary_table(uva_summary))
            
            if uva_removed:
                story.append(Spacer(1, 0.3*inch))
                story.append(Paragraph("Items Removed by UVA:", styles['Heading2']))
                
                confirmed_items = st.session_state.get("previous_items", [])
                removed_items_text = []
                for item_label, wto_score in uva_removed:
                    actual_text = get_item_text_from_label(item_label, confirmed_items)
                    removed_items_text.append(f"{actual_text} (wTO: {wto_score:.4f})")
                
                removed_table = create_items_table(removed_items_text, "Removed Items")
                story.append(removed_table)
        
        story.append(PageBreak())
        
        # ====================
        # BOOTEGA RESULTS
        # ====================
        if st.session_state.get("bootega_status") == "Completed":
            story.append(Paragraph("Bootstrap EGA Stability Analysis", styles['Heading1']))
            
            stable_items = st.session_state.get("bootega_stable_items", [])
            removed_log = st.session_state.get("bootega_removed_log", [])
            stability_scores = st.session_state.get("bootega_final_stability_scores", {})
            stability_threshold = st.session_state.get("bootega_stability_threshold", 0.75)
            n_bootstraps = st.session_state.get("bootega_n_bootstraps", 100)
            
            story.append(Paragraph(f"Bootstrap stability analysis was conducted with {n_bootstraps} bootstrap iterations "
                                 f"and a stability threshold of {stability_threshold:.2f}. Items failing to meet "
                                 f"the stability criterion were iteratively removed.",
                                 styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
            
            # NMI comparison
            initial_nmi = st.session_state.get("bootega_initial_nmi_compared", np.nan)
            final_nmi = st.session_state.get("bootega_final_nmi", np.nan)
            
            bootega_summary = {
                'Bootstrap Iterations': n_bootstraps,
                'Stability Threshold': f"{stability_threshold:.2f}",
                'Items Removed': len(removed_log),
                'Final Stable Items': len(stable_items),
                'Initial NMI': f"{initial_nmi:.4f}" if not pd.isna(initial_nmi) else "N/A",
                'Final NMI': f"{final_nmi:.4f}" if not pd.isna(final_nmi) else "N/A"
            }
            
            if not pd.isna(initial_nmi) and not pd.isna(final_nmi):
                bootega_summary['NMI Improvement'] = f"{final_nmi - initial_nmi:+.4f}"
            
            story.append(create_summary_table(bootega_summary))
            
            # Final stable items
            if stable_items:
                story.append(Spacer(1, 0.3*inch))
                story.append(Paragraph("Final Stable Items:", styles['Heading2']))
                
                confirmed_items = st.session_state.get("previous_items", [])
                stable_items_text = []
                for item_label in stable_items:
                    actual_text = get_item_text_from_label(item_label, confirmed_items)
                    stability_score = stability_scores.get(item_label, np.nan)
                    score_text = f" (Stability: {stability_score:.4f})" if not pd.isna(stability_score) else ""
                    stable_items_text.append(f"{actual_text}{score_text}")
                
                stable_table = create_items_table(stable_items_text, "Final Items")
                story.append(stable_table)
        
        story.append(PageBreak())
        
        # ====================
        # TECHNICAL APPENDIX
        # ====================
        story.append(Paragraph("Technical Appendix", styles['Heading1']))
        
        # Configuration parameters
        story.append(Paragraph("Analysis Configuration", styles['Heading2']))
        
        config_data = {
            'Focus Area': st.session_state.get("assessment_topic", "Unknown"),
            'LLM Model': "GPT-4o",
            'Embedding Model': "text-embedding-3-small",
            'Network Method': st.session_state.get("network_method_select", "Unknown"),
            'Embedding Type': st.session_state.get("input_matrix_select", "Unknown"),
            'UVA Threshold': f"{st.session_state.get('uva_threshold', 0.20):.2f}",
            'bootEGA Bootstraps': st.session_state.get("bootega_n_bootstraps", "N/A"),
            'Stability Threshold': f"{st.session_state.get('bootega_stability_threshold', 0.75):.2f}",
            'Parallel Processing': "Enabled" if st.session_state.get("bootega_use_parallel", True) else "Disabled"
        }
        
        story.append(create_summary_table(config_data))
        
        # Software information
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("Software Information", styles['Heading2']))
        
        software_text = """
        <b>SLTItemGen Version:</b> 0.1.0 (Development)<br/>
        <b>Python Libraries:</b> OpenAI, NetworkX, igraph, scikit-learn, pandas, matplotlib<br/>
        <b>AI-GENIE Implementation:</b> Pure Python (adapted from van Lissa et al., 2024)<br/>
        <b>Report Generated:</b> """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        
        story.append(Paragraph(software_text, styles['Code']))
        
        # Build PDF
        doc.build(story)
        
        # Read the generated PDF
        with open(temp_filename, 'rb') as f:
            pdf_content = f.read()
            
        return pdf_content
        
    finally:
        # Clean up temporary file
        try:
            import os
            os.unlink(temp_filename)
        except:
            pass  # Ignore cleanup errors 