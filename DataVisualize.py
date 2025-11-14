"""
DataVisualize.py - Visualization Module
Hiển thị các bước xử lý text và retrieval process
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
from pathlib import Path

class DataVisualizer:
    """Visualizer cho text processing và retrieval"""

    def __init__(self, save_dir: str = "visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')

    def visualize_text_processing(self, stats: Dict, save: bool = True):
        """
        Visualize text processing steps

        Args:
            stats: Dictionary chứa statistics từ TextProcessor
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Token count through processing steps
        steps = ['Original', 'After\nNormalization', 'After\nStopping']
        counts = [
            stats.get('original_tokens', 0),
            stats.get('after_normalization', 0),
            stats.get('after_stopping', 0)
        ]

        colors = ['#3498db', '#2ecc71', '#e74c3c']
        bars = ax1.bar(steps, counts, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Number of Tokens', fontsize=12, fontweight='bold')
        ax1.set_title('Text Processing Pipeline', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontweight='bold')

        # Plot 2: Pie chart showing token reduction
        removed = stats.get('stopwords_removed', 0)
        kept = stats.get('after_stopping', 0)

        if removed + kept > 0:
            sizes = [kept, removed]
            labels = [f'Kept\n({kept} tokens)', f'Removed\n({removed} tokens)']
            colors_pie = ['#2ecc71', '#e74c3c']
            explode = (0.05, 0)

            ax2.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                   autopct='%1.1f%%', shadow=True, startangle=90)
            ax2.set_title('Stopword Removal Impact', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save:
            save_path = self.save_dir / "text_processing.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved visualization: {save_path}")

        plt.close()
        return fig

    def visualize_retrieval_stages(self, stats: Dict, save: bool = True):
        """
        Visualize three-stage retrieval process

        Args:
            stats: Dictionary chứa statistics từ ThreeStageRetrieval
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Number of candidates through stages
        stages = ['Stage 1\n(BM25)', 'Stage 2\n(Embedding)', 'Stage 3\n(Reranking)']
        candidates = [
            len(stats.get('stage1_results', [])),
            len(stats.get('stage2_results', [])),
            len(stats.get('stage3_results', []))
        ]

        colors = ['#3498db', '#9b59b6', '#e74c3c']
        bars = ax1.bar(stages, candidates, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Number of Candidates', fontsize=12, fontweight='bold')
        ax1.set_title('Three-Stage Retrieval Pipeline', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontweight='bold')

        # Plot 2: Processing time for each stage
        times = [
            stats.get('stage1_time', 0) * 1000,  # Convert to ms
            stats.get('stage2_time', 0) * 1000,
            stats.get('stage3_time', 0) * 1000
        ]

        bars2 = ax2.bar(stages, times, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Processing Time (ms)', fontsize=12, fontweight='bold')
        ax2.set_title('Stage Processing Times', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}ms',
                        ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        if save:
            save_path = self.save_dir / "retrieval_stages.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved visualization: {save_path}")

        plt.close()
        return fig

    def visualize_score_distribution(self, 
                                     stage_results: List[tuple],
                                     stage_name: str,
                                     save: bool = True):
        """
        Visualize score distribution for a stage

        Args:
            stage_results: List of (doc_id, score) tuples
            stage_name: Name of the stage
        """
        if not stage_results:
            return None

        fig, ax = plt.subplots(figsize=(10, 5))

        # Extract scores
        scores = [score for _, score in stage_results]
        doc_ids = [f"Doc {doc_id}" for doc_id, _ in stage_results[:20]]  # Top 20

        # Plot
        colors = plt.cm.viridis(np.linspace(0, 1, len(scores[:20])))
        bars = ax.barh(doc_ids[:20], scores[:20], color=colors, edgecolor='black')

        ax.set_xlabel('Relevance Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Documents', fontsize=12, fontweight='bold')
        ax.set_title(f'{stage_name} - Top 20 Documents', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        if save:
            filename = f"{stage_name.lower().replace(' ', '_')}_scores.png"
            save_path = self.save_dir / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved visualization: {save_path}")

        plt.close()
        return fig

    def create_summary_report(self, 
                             text_stats: Dict,
                             retrieval_stats: Dict,
                             save: bool = True):
        """
        Create a comprehensive summary visualization
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. Text Processing
        ax1 = fig.add_subplot(gs[0, 0])
        steps = ['Original', 'Normalized', 'Filtered']
        counts = [
            text_stats.get('original_tokens', 0),
            text_stats.get('after_normalization', 0),
            text_stats.get('after_stopping', 0)
        ]
        ax1.bar(steps, counts, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.7)
        ax1.set_title('Text Processing', fontweight='bold')
        ax1.set_ylabel('Token Count')
        ax1.grid(axis='y', alpha=0.3)

        # 2. Retrieval Stages
        ax2 = fig.add_subplot(gs[0, 1])
        stages = ['BM25', 'Embedding', 'Reranking']
        candidates = [
            len(retrieval_stats.get('stage1_results', [])),
            len(retrieval_stats.get('stage2_results', [])),
            len(retrieval_stats.get('stage3_results', []))
        ]
        ax2.bar(stages, candidates, color=['#3498db', '#9b59b6', '#e74c3c'], alpha=0.7)
        ax2.set_title('Retrieval Pipeline', fontweight='bold')
        ax2.set_ylabel('Candidates')
        ax2.grid(axis='y', alpha=0.3)

        # 3. Processing Times
        ax3 = fig.add_subplot(gs[1, :])
        times = [
            retrieval_stats.get('stage1_time', 0) * 1000,
            retrieval_stats.get('stage2_time', 0) * 1000,
            retrieval_stats.get('stage3_time', 0) * 1000
        ]
        ax3.bar(stages, times, color=['#3498db', '#9b59b6', '#e74c3c'], alpha=0.7)
        ax3.set_title('Stage Processing Times', fontweight='bold')
        ax3.set_ylabel('Time (ms)')
        ax3.grid(axis='y', alpha=0.3)

        # 4. Summary Statistics
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')

        summary_text = f"""
        PROCESSING SUMMARY

        Text Processing:
        • Original tokens: {text_stats.get('original_tokens', 0)}
        • After normalization: {text_stats.get('after_normalization', 0)}
        • After stopping: {text_stats.get('after_stopping', 0)}
        • Unique tokens: {text_stats.get('unique_tokens', 0)}

        Retrieval Process:
        • Stage 1 (BM25): {len(retrieval_stats.get('stage1_results', []))} candidates in {retrieval_stats.get('stage1_time', 0):.3f}s
        • Stage 2 (Embedding): {len(retrieval_stats.get('stage2_results', []))} candidates in {retrieval_stats.get('stage2_time', 0):.3f}s
        • Stage 3 (Reranking): {len(retrieval_stats.get('stage3_results', []))} results in {retrieval_stats.get('stage3_time', 0):.3f}s

        Total Time: {sum([retrieval_stats.get('stage1_time', 0), retrieval_stats.get('stage2_time', 0), retrieval_stats.get('stage3_time', 0)]):.3f}s
        """

        ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center')

        if save:
            save_path = self.save_dir / "summary_report.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved summary report: {save_path}")

        plt.close()
        return fig
