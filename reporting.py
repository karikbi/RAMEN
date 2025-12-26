#!/usr/bin/env python3
"""
Wallpaper Curation Pipeline - Part 3: Reporting

Generates comprehensive statistics and reports for the curation pipeline.
"""

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("wallpaper_curator")


@dataclass
class PipelineStats:
    """Complete pipeline statistics."""
    # Timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    fetch_duration_sec: float = 0.0
    filter_duration_sec: float = 0.0
    embed_duration_sec: float = 0.0
    upload_duration_sec: float = 0.0
    
    # Candidates by source
    reddit_candidates: int = 0
    unsplash_candidates: int = 0
    pexels_candidates: int = 0
    total_candidates: int = 0
    
    # Filter results
    passed_hard_filters: int = 0
    rejected_resolution: int = 0
    rejected_file_integrity: int = 0
    rejected_aspect_ratio: int = 0
    rejected_text_detection: int = 0
    rejected_file_size: int = 0
    rejected_duplicate: int = 0
    
    # Quality scoring
    passed_quality_scoring: int = 0
    rejected_quality_score: int = 0
    
    # Final results
    approved_count: int = 0
    uploaded_count: int = 0
    upload_failures: int = 0
    
    # Quality metrics
    quality_scores: list[float] = field(default_factory=list)
    
    # Category distribution
    category_counts: dict[str, int] = field(default_factory=dict)
    
    # Source success rates
    source_results: dict[str, dict] = field(default_factory=dict)


class ReportGenerator:
    """Generates comprehensive pipeline reports."""
    
    def __init__(self, output_dir: Path = Path("./reports")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _calculate_histogram(
        self,
        values: list[float],
        bins: int = 10,
        min_val: float = 0.0,
        max_val: float = 1.0
    ) -> list[dict]:
        """Calculate histogram for a list of values."""
        if not values:
            return []
        
        bin_width = (max_val - min_val) / bins
        histogram = []
        
        for i in range(bins):
            bin_start = min_val + i * bin_width
            bin_end = bin_start + bin_width
            count = sum(1 for v in values if bin_start <= v < bin_end)
            histogram.append({
                "range": f"{bin_start:.2f}-{bin_end:.2f}",
                "count": count
            })
        
        return histogram
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def generate_report(self, stats: PipelineStats) -> dict[str, Any]:
        """
        Generate comprehensive report from pipeline statistics.
        
        Args:
            stats: PipelineStats object with all metrics.
        
        Returns:
            Report as dictionary.
        """
        # Calculate derived metrics
        total_rejected = (
            stats.rejected_resolution +
            stats.rejected_file_integrity +
            stats.rejected_aspect_ratio +
            stats.rejected_text_detection +
            stats.rejected_file_size +
            stats.rejected_duplicate +
            stats.rejected_quality_score
        )
        
        avg_quality = (
            sum(stats.quality_scores) / len(stats.quality_scores)
            if stats.quality_scores else 0.0
        )
        
        total_duration = (
            stats.fetch_duration_sec +
            stats.filter_duration_sec +
            stats.embed_duration_sec +
            stats.upload_duration_sec
        )
        
        # Build report
        report = {
            "report_generated": datetime.now().isoformat() + "Z",
            "pipeline_run": {
                "start": stats.start_time.isoformat() if stats.start_time else None,
                "end": stats.end_time.isoformat() if stats.end_time else None,
                "total_duration": self._format_duration(total_duration),
                "stage_durations": {
                    "fetching": self._format_duration(stats.fetch_duration_sec),
                    "filtering": self._format_duration(stats.filter_duration_sec),
                    "embeddings": self._format_duration(stats.embed_duration_sec),
                    "uploading": self._format_duration(stats.upload_duration_sec),
                }
            },
            "candidates": {
                "total": stats.total_candidates,
                "by_source": {
                    "reddit": stats.reddit_candidates,
                    "unsplash": stats.unsplash_candidates,
                    "pexels": stats.pexels_candidates,
                }
            },
            "filtering": {
                "passed_hard_filters": stats.passed_hard_filters,
                "passed_quality_scoring": stats.passed_quality_scoring,
                "total_rejected": total_rejected,
                "rejection_breakdown": {
                    "resolution": stats.rejected_resolution,
                    "file_integrity": stats.rejected_file_integrity,
                    "aspect_ratio": stats.rejected_aspect_ratio,
                    "text_detection": stats.rejected_text_detection,
                    "file_size": stats.rejected_file_size,
                    "duplicate": stats.rejected_duplicate,
                    "quality_score": stats.rejected_quality_score,
                }
            },
            "quality": {
                "average_score": round(avg_quality, 4),
                "min_score": round(min(stats.quality_scores), 4) if stats.quality_scores else 0,
                "max_score": round(max(stats.quality_scores), 4) if stats.quality_scores else 0,
                "score_distribution": self._calculate_histogram(stats.quality_scores),
            },
            "results": {
                "approved": stats.approved_count,
                "uploaded": stats.uploaded_count,
                "upload_failures": stats.upload_failures,
                "approval_rate": round(
                    stats.approved_count / stats.total_candidates * 100
                    if stats.total_candidates > 0 else 0, 2
                ),
            },
            "categories": stats.category_counts,
            "source_performance": {}
        }
        
        # Add source performance metrics
        for source, data in stats.source_results.items():
            candidates = data.get("candidates", 0)
            approved = data.get("approved", 0)
            rate = approved / candidates * 100 if candidates > 0 else 0
            
            report["source_performance"][source] = {
                "candidates": candidates,
                "approved": approved,
                "success_rate": round(rate, 2),
                "recommendation": self._get_recommendation(rate)
            }
        
        return report
    
    def _get_recommendation(self, rate: float) -> str:
        """Get recommendation based on success rate."""
        if rate >= 30:
            return "HIGH_PERFORMER - Consider increasing allocation"
        elif rate >= 10:
            return "GOOD - Maintain current allocation"
        elif rate >= 5:
            return "LOW - Consider reducing allocation"
        else:
            return "POOR - Consider removing source"
    
    def save_report(
        self,
        report: dict[str, Any],
        filename: str = None
    ) -> Path:
        """Save report to JSON file."""
        if filename is None:
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{date_str}.json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved report to {output_path}")
        return output_path
    
    def print_summary(self, report: dict[str, Any]) -> None:
        """Print formatted summary to console and log."""
        lines = [
            "",
            "=" * 70,
            "PIPELINE EXECUTION REPORT",
            "=" * 70,
            "",
            "📊 SUMMARY",
            "-" * 70,
            f"  Total Candidates:     {report['candidates']['total']}",
            f"    - Reddit:           {report['candidates']['by_source']['reddit']}",
            f"    - Unsplash:         {report['candidates']['by_source']['unsplash']}",
            f"    - Pexels:           {report['candidates']['by_source']['pexels']}",
            "",
            f"  Passed Hard Filters:  {report['filtering']['passed_hard_filters']}",
            f"  Passed Quality Score: {report['filtering']['passed_quality_scoring']}",
            f"  Final Approved:       {report['results']['approved']}",
            f"  Upload Success:       {report['results']['uploaded']}",
            "",
            "🏆 QUALITY METRICS",
            "-" * 70,
            f"  Average Score:        {report['quality']['average_score']:.4f}",
            f"  Score Range:          {report['quality']['min_score']:.4f} - {report['quality']['max_score']:.4f}",
            f"  Approval Rate:        {report['results']['approval_rate']:.1f}%",
            "",
            "❌ REJECTION BREAKDOWN",
            "-" * 70,
        ]
        
        breakdown = report['filtering']['rejection_breakdown']
        for reason, count in breakdown.items():
            lines.append(f"  {reason.replace('_', ' ').title():20} {count}")
        
        lines.extend([
            "",
            "📁 CATEGORY DISTRIBUTION",
            "-" * 70,
        ])
        
        for category, count in sorted(
            report['categories'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            lines.append(f"  {category:20} {count}")
        
        lines.extend([
            "",
            "⏱️  TIMING",
            "-" * 70,
            f"  Total Duration:       {report['pipeline_run']['total_duration']}",
            f"    - Fetching:         {report['pipeline_run']['stage_durations']['fetching']}",
            f"    - Filtering:        {report['pipeline_run']['stage_durations']['filtering']}",
            f"    - Embeddings:       {report['pipeline_run']['stage_durations']['embeddings']}",
            f"    - Uploading:        {report['pipeline_run']['stage_durations']['uploading']}",
            "",
            "📈 SOURCE PERFORMANCE",
            "-" * 70,
        ])
        
        for source, perf in report['source_performance'].items():
            lines.append(
                f"  {source:20} {perf['success_rate']:5.1f}% ({perf['approved']}/{perf['candidates']}) - {perf['recommendation']}"
            )
        
        lines.extend([
            "",
            "=" * 70,
        ])
        
        # Print and log
        output = "\n".join(lines)
        print(output)
        logger.info(output)


def generate_report(
    stats: PipelineStats,
    output_dir: Path = Path("./reports"),
    print_summary: bool = True
) -> Path:
    """
    Generate and save pipeline report.
    
    Args:
        stats: PipelineStats with all metrics.
        output_dir: Directory to save report.
        print_summary: Whether to print summary to console.
    
    Returns:
        Path to saved report file.
    """
    generator = ReportGenerator(output_dir)
    report = generator.generate_report(stats)
    
    if print_summary:
        generator.print_summary(report)
    
    # Save detailed report
    report_path = generator.save_report(report)
    
    # Also save simplified statistics.json for GitHub Actions workflow
    save_statistics(stats, output_dir)
    
    return report_path


def save_statistics(stats: PipelineStats, output_dir: Path = Path("./reports")) -> Path:
    """
    Save simplified statistics.json for GitHub Actions workflow.
    
    The workflow expects these specific field names:
    - candidates_total
    - approved_total  
    - rejected_total
    
    Args:
        stats: PipelineStats with all metrics.
        output_dir: Directory to save statistics.
    
    Returns:
        Path to statistics.json file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate total rejected
    total_rejected = (
        stats.rejected_resolution +
        stats.rejected_file_integrity +
        stats.rejected_aspect_ratio +
        stats.rejected_text_detection +
        stats.rejected_file_size +
        stats.rejected_duplicate +
        stats.rejected_quality_score
    )
    
    statistics = {
        "candidates_total": stats.total_candidates,
        "approved_total": stats.approved_count,
        "rejected_total": total_rejected,
        "uploaded_total": stats.uploaded_count,
        # Additional fields for debugging
        "by_source": {
            "reddit": stats.reddit_candidates,
            "unsplash": stats.unsplash_candidates,
            "pexels": stats.pexels_candidates,
        },
        "passed_hard_filters": stats.passed_hard_filters,
        "passed_quality_scoring": stats.passed_quality_scoring,
        "rejection_breakdown": {
            "resolution": stats.rejected_resolution,
            "file_integrity": stats.rejected_file_integrity,
            "aspect_ratio": stats.rejected_aspect_ratio,
            "text_detection": stats.rejected_text_detection,
            "file_size": stats.rejected_file_size,
            "duplicate": stats.rejected_duplicate,
            "quality_score": stats.rejected_quality_score,
        },
        "timestamp": datetime.now().isoformat() + "Z",
    }
    
    output_path = output_dir / "statistics.json"
    
    with open(output_path, "w") as f:
        json.dump(statistics, f, indent=2)
    
    logger.info(f"Saved statistics to {output_path}")
    return output_path
