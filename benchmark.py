#!/usr/bin/env python3
"""
RAMEN Pipeline - Performance Benchmarking

Measures performance of each pipeline stage and identifies bottlenecks.
Generates benchmark reports for tracking performance over time.

Usage:
    python benchmark.py --all
    python benchmark.py --stages fetch,filter
    python benchmark.py --compare-baseline benchmarks/baseline.json
"""

import argparse
import asyncio
import gc
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import psutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("benchmark")


@dataclass
class BenchmarkResult:
    """Result of a single benchmark measurement."""
    stage: str
    duration_ms: float
    memory_mb: float
    items_processed: int
    items_per_second: float
    peak_memory_mb: float = 0.0
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "duration_ms": round(self.duration_ms, 2),
            "memory_mb": round(self.memory_mb, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "items_processed": self.items_processed,
            "items_per_second": round(self.items_per_second, 3),
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    timestamp: str
    python_version: str
    system_info: dict
    results: list[BenchmarkResult]
    total_duration_ms: float
    notes: str = ""
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "python_version": self.python_version,
            "system_info": self.system_info,
            "results": [r.to_dict() for r in self.results],
            "total_duration_ms": round(self.total_duration_ms, 2),
            "notes": self.notes,
        }


class MemoryTracker:
    """Track memory usage during benchmark."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.peak_memory = 0
        self._initial = 0
    
    def start(self):
        """Start tracking."""
        gc.collect()
        self._initial = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory = self._initial
    
    def update(self):
        """Update peak memory."""
        current = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory = max(self.peak_memory, current)
    
    def get_current(self) -> float:
        """Get current memory in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_delta(self) -> float:
        """Get memory change since start."""
        return self.get_current() - self._initial


class PipelineBenchmark:
    """
    Performance benchmarking for pipeline stages.
    
    Measures:
    - Execution time per stage
    - Memory usage
    - Throughput (items/second)
    - Bottleneck identification
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.memory_tracker = MemoryTracker()
        self.results: list[BenchmarkResult] = []
    
    def _get_system_info(self) -> dict:
        """Get system information."""
        import platform
        import sys
        
        return {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "memory_total_gb": round(psutil.virtual_memory().total / 1024 / 1024 / 1024, 2),
            "python_version": sys.version,
        }
    
    def _log(self, message: str):
        """Log if verbose."""
        if self.verbose:
            logger.info(message)
    
    def time_function(
        self,
        func: Callable,
        stage_name: str,
        items_count: int = 1,
        **kwargs
    ) -> BenchmarkResult:
        """
        Time a synchronous function.
        
        Args:
            func: Function to benchmark
            stage_name: Name for this benchmark
            items_count: Number of items processed
            **kwargs: Arguments to pass to function
        """
        self._log(f"⏱️  Benchmarking: {stage_name}")
        
        self.memory_tracker.start()
        start_time = time.perf_counter()
        
        try:
            result = func(**kwargs)
        finally:
            end_time = time.perf_counter()
            self.memory_tracker.update()
        
        duration_ms = (end_time - start_time) * 1000
        memory_mb = self.memory_tracker.get_delta()
        items_per_sec = items_count / (duration_ms / 1000) if duration_ms > 0 else 0
        
        benchmark_result = BenchmarkResult(
            stage=stage_name,
            duration_ms=duration_ms,
            memory_mb=memory_mb,
            peak_memory_mb=self.memory_tracker.peak_memory,
            items_processed=items_count,
            items_per_second=items_per_sec,
        )
        
        self.results.append(benchmark_result)
        self._log(f"   Duration: {duration_ms:.2f}ms, Memory: {memory_mb:.2f}MB")
        
        return benchmark_result
    
    async def time_async_function(
        self,
        func: Callable,
        stage_name: str,
        items_count: int = 1,
        **kwargs
    ) -> BenchmarkResult:
        """Time an async function."""
        self._log(f"⏱️  Benchmarking: {stage_name}")
        
        self.memory_tracker.start()
        start_time = time.perf_counter()
        
        try:
            result = await func(**kwargs)
        finally:
            end_time = time.perf_counter()
            self.memory_tracker.update()
        
        duration_ms = (end_time - start_time) * 1000
        memory_mb = self.memory_tracker.get_delta()
        items_per_sec = items_count / (duration_ms / 1000) if duration_ms > 0 else 0
        
        benchmark_result = BenchmarkResult(
            stage=stage_name,
            duration_ms=duration_ms,
            memory_mb=memory_mb,
            peak_memory_mb=self.memory_tracker.peak_memory,
            items_processed=items_count,
            items_per_second=items_per_sec,
        )
        
        self.results.append(benchmark_result)
        self._log(f"   Duration: {duration_ms:.2f}ms, Memory: {memory_mb:.2f}MB")
        
        return benchmark_result
    
    def benchmark_filters(self, image_paths: list[Path]) -> BenchmarkResult:
        """Benchmark filtering stage."""
        from filters import HardFilters, FilterConfig
        
        filters = HardFilters(FilterConfig())
        
        def run_filters():
            results = []
            for i, path in enumerate(image_paths):
                result = filters.apply_all_filters(path, f"test_{i}", source="benchmark")
                results.append(result)
            return results
        
        return self.time_function(
            run_filters,
            "Hard Filters",
            items_count=len(image_paths)
        )
    
    def benchmark_quality_scoring(self, image_paths: list[Path]) -> BenchmarkResult:
        """Benchmark quality scoring (using ML-based scorer)."""
        from ml_quality_scorer import MLQualityScorer
        
        scorer = MLQualityScorer()
        
        def run_scoring():
            scores = []
            for path in image_paths:
                score = scorer.score(path)
                scores.append(score)
            return scores
        
        return self.time_function(
            run_scoring,
            "Quality Scoring (ML)",
            items_count=len(image_paths)
        )

    
    def benchmark_metadata_generation(self, image_paths: list[Path]) -> BenchmarkResult:
        """Benchmark metadata generation."""
        from metadata_generator import MetadataGenerator
        
        generator = MetadataGenerator()
        
        def run_metadata():
            metadata_list = []
            for i, path in enumerate(image_paths):
                metadata = generator.generate_metadata(
                    filepath=path,
                    title=f"Test Wallpaper {i}",
                    artist="Test Artist",
                    source="benchmark",
                    source_metadata={},
                    quality_score=0.9
                )
                metadata_list.append(metadata)
            return metadata_list
        
        return self.time_function(
            run_metadata,
            "Metadata Generation",
            items_count=len(image_paths)
        )
    
    def benchmark_embeddings(self, image_paths: list[Path], models: list[str] = None) -> list[BenchmarkResult]:
        """Benchmark embedding extraction per model."""
        try:
            from embeddings import EmbeddingExtractor
        except ImportError:
            logger.warning("Embeddings module requires ML dependencies, skipping")
            return []
        
        if models is None:
            models = ["mobilenet", "efficientnet", "siglip", "dinov2"]
        
        extractor = EmbeddingExtractor()
        results = []
        
        model_funcs = {
            "mobilenet": extractor.extract_mobilenet,
            "efficientnet": extractor.extract_efficientnet,
            "siglip": extractor.extract_siglip,
            "dinov2": extractor.extract_dinov2,
        }
        
        for model_name in models:
            if model_name not in model_funcs:
                continue
            
            func = model_funcs[model_name]
            
            def run_model():
                embeddings = []
                for path in image_paths:
                    emb = func(path)
                    embeddings.append(emb)
                return embeddings
            
            result = self.time_function(
                run_model,
                f"Embedding: {model_name}",
                items_count=len(image_paths)
            )
            results.append(result)
        
        return results
    
    def generate_report(self, notes: str = "") -> BenchmarkReport:
        """Generate complete benchmark report."""
        import sys
        
        total_duration = sum(r.duration_ms for r in self.results)
        
        return BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            python_version=sys.version,
            system_info=self._get_system_info(),
            results=self.results,
            total_duration_ms=total_duration,
            notes=notes,
        )
    
    def compare_with_baseline(
        self,
        current: BenchmarkReport,
        baseline_path: Path
    ) -> dict:
        """Compare current results with baseline."""
        if not baseline_path.exists():
            return {"error": f"Baseline not found: {baseline_path}"}
        
        with open(baseline_path) as f:
            baseline = json.load(f)
        
        comparisons = []
        baseline_results = {r["stage"]: r for r in baseline.get("results", [])}
        
        for result in current.results:
            stage = result.stage
            if stage in baseline_results:
                base = baseline_results[stage]
                
                time_change = (result.duration_ms - base["duration_ms"]) / base["duration_ms"] * 100
                memory_change = (result.memory_mb - base["memory_mb"]) / max(base["memory_mb"], 0.1) * 100
                
                comparisons.append({
                    "stage": stage,
                    "time_change_percent": round(time_change, 1),
                    "memory_change_percent": round(memory_change, 1),
                    "regression": time_change > 20,  # >20% slower is regression
                })
        
        return {
            "baseline_date": baseline.get("timestamp", "unknown"),
            "comparisons": comparisons,
            "regressions_found": any(c["regression"] for c in comparisons),
        }
    
    def save_report(self, report: BenchmarkReport, output_path: Path) -> None:
        """Save benchmark report to JSON file."""
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Report saved to {output_path}")
    
    def print_report(self, report: BenchmarkReport) -> None:
        """Print formatted benchmark report."""
        print("\n" + "=" * 70)
        print("RAMEN PIPELINE BENCHMARK REPORT")
        print("=" * 70)
        print(f"Timestamp: {report.timestamp}")
        print(f"System: {report.system_info.get('platform', 'Unknown')}")
        print(f"CPUs: {report.system_info.get('cpu_count', 'Unknown')}")
        print(f"Memory: {report.system_info.get('memory_total_gb', 'Unknown')} GB")
        print("-" * 70)
        
        print(f"\n{'Stage':<30} {'Time (ms)':<12} {'Memory (MB)':<12} {'Items/sec':<10}")
        print("-" * 70)
        
        for result in report.results:
            print(f"{result.stage:<30} {result.duration_ms:<12.2f} {result.memory_mb:<12.2f} {result.items_per_second:<10.2f}")
        
        print("-" * 70)
        print(f"{'TOTAL':<30} {report.total_duration_ms:<12.2f}")
        
        # Identify bottleneck
        if report.results:
            slowest = max(report.results, key=lambda r: r.duration_ms)
            pct = slowest.duration_ms / report.total_duration_ms * 100
            print(f"\n⚠️  Bottleneck: {slowest.stage} ({pct:.1f}% of total time)")
        
        if report.notes:
            print(f"\nNotes: {report.notes}")
        
        print("=" * 70)


def create_test_images(n: int = 10, output_dir: Path = Path("./temp/benchmark")) -> list[Path]:
    """Create test images for benchmarking."""
    from PIL import Image
    import random
    
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    
    for i in range(n):
        # Create random colored image at 2560x1440
        width, height = 2560, 1440
        img = Image.new("RGB", (width, height))
        
        # Add some variation (gradient + noise)
        pixels = img.load()
        r_base = random.randint(0, 200)
        g_base = random.randint(0, 200)
        b_base = random.randint(0, 200)
        
        for y in range(height):
            for x in range(width):
                r = min(255, r_base + int(x / width * 55) + random.randint(-10, 10))
                g = min(255, g_base + int(y / height * 55) + random.randint(-10, 10))
                b = min(255, b_base + random.randint(-10, 10))
                pixels[x, y] = (r, g, b)
        
        path = output_dir / f"test_wallpaper_{i:04d}.jpg"
        img.save(path, "JPEG", quality=90)
        paths.append(path)
    
    return paths


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RAMEN pipeline performance"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks"
    )
    parser.add_argument(
        "--stages",
        type=str,
        help="Comma-separated list of stages to benchmark"
    )
    parser.add_argument(
        "--images",
        type=int,
        default=10,
        help="Number of test images to use"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./benchmarks/benchmark_report.json"),
        help="Output path for report"
    )
    parser.add_argument(
        "--compare-baseline",
        type=Path,
        help="Compare with baseline report"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output"
    )
    
    args = parser.parse_args()
    
    benchmark = PipelineBenchmark(verbose=not args.quiet)
    
    # Determine stages to run
    if args.stages:
        stages = args.stages.split(",")
    else:
        stages = ["filters", "quality", "metadata"]  # Skip embeddings by default (slow)
    
    # Create test images
    print(f"Creating {args.images} test images...")
    test_images = create_test_images(args.images)
    
    try:
        # Run benchmarks
        if "filters" in stages:
            benchmark.benchmark_filters(test_images)
        
        if "quality" in stages:
            benchmark.benchmark_quality_scoring(test_images)
        
        if "metadata" in stages:
            benchmark.benchmark_metadata_generation(test_images)
        
        if "embeddings" in stages or args.all:
            benchmark.benchmark_embeddings(test_images)
        
        # Generate report
        report = benchmark.generate_report(notes=f"Test images: {args.images}")
        
        if not args.quiet:
            benchmark.print_report(report)
        
        # Save report
        args.output.parent.mkdir(parents=True, exist_ok=True)
        benchmark.save_report(report, args.output)
        
        # Compare with baseline if provided
        if args.compare_baseline:
            comparison = benchmark.compare_with_baseline(report, args.compare_baseline)
            print("\nBaseline Comparison:")
            print(json.dumps(comparison, indent=2))
            
            if comparison.get("regressions_found"):
                print("\n⚠️  Performance regressions detected!")
                return 1
    
    finally:
        # Cleanup test images
        for path in test_images:
            try:
                path.unlink()
            except:
                pass
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
