#!/usr/bin/env python3
"""
RAMEN Pipeline - Collection Validator

Validates the health and integrity of the wallpaper collection.
Run this weekly via GitHub Actions for ongoing monitoring.

Usage:
    python validate_collection.py --full-check
    python validate_collection.py --quick-check
    python validate_collection.py --fix-issues
"""

import argparse
import asyncio
import gzip
import json
import logging
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiohttp

# Import central config
try:
    from config_loader import get_config
    HAS_CONFIG_LOADER = True
except ImportError:
    HAS_CONFIG_LOADER = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("validator")


@dataclass
class ValidationResult:
    """Result of a validation check."""
    check_name: str
    passed: bool
    message: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report."""
    timestamp: str
    overall_passed: bool
    checks: list[ValidationResult] = field(default_factory=list)
    total_wallpapers: int = 0
    summary: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "overall_passed": self.overall_passed,
            "total_wallpapers": self.total_wallpapers,
            "checks": [
                {
                    "name": c.check_name,
                    "passed": c.passed,
                    "message": c.message,
                    "errors": c.errors,
                    "warnings": c.warnings,
                    "stats": c.stats,
                }
                for c in self.checks
            ],
            "summary": self.summary,
        }


class CollectionValidator:
    """
    Validates wallpaper collection health and integrity.
    
    Checks:
    - Manifest JSON structure and required fields
    - Embedding dimensions correctness
    - Duplicate ID detection
    - R2 URL accessibility (spot check)
    - Quality score validation
    - Category distribution balance
    - Perceptual hash collisions
    """
    
    # Required fields with aliases (field, [aliases]) - any of the names is valid
    REQUIRED_FIELDS = [
        ("id", []),
        ("title", []),
        ("r2_url", ["url"]),  # Accept 'url' as alias for 'r2_url'
        ("quality_score", []),
        ("primary_category", ["category"]),  # Accept 'category' as alias
        ("date_added", []),
    ]
    
    EMBEDDING_DIMENSIONS = {
        "mobilenet_v3": 576,
        "efficientnet_v2": 1280,
        "siglip": 1152,
        "dinov2": 1024,
    }
    
    def __init__(
        self,
        manifests_dir: Path = Path("./manifests"),
        collection_file: str = "collection.json"
    ):
        self.manifests_dir = manifests_dir
        self.collection_file = collection_file
        self.collection: list[dict] = []
    
    def load_collection(self) -> bool:
        """Load collection from manifest file."""
        manifest_path = self.manifests_dir / self.collection_file
        
        # Try compressed first
        gz_path = manifest_path.with_suffix(".json.gz")
        if gz_path.exists():
            with gzip.open(gz_path, "rt", encoding="utf-8") as f:
                data = json.load(f)
                # Handle both list format and wrapped format {"wallpapers": [...]}
                if isinstance(data, list):
                    self.collection = data
                elif isinstance(data, dict):
                    self.collection = data.get("wallpapers", [])
                else:
                    logger.error(f"Unexpected manifest format: {type(data)}")
                    return False
            logger.info(f"Loaded {len(self.collection)} wallpapers from {gz_path}")
            return True
        
        # Try uncompressed
        if manifest_path.exists():
            with open(manifest_path) as f:
                data = json.load(f)
                # Handle both list format and wrapped format {"wallpapers": [...]}
                if isinstance(data, list):
                    self.collection = data
                elif isinstance(data, dict):
                    self.collection = data.get("wallpapers", [])
                else:
                    logger.error(f"Unexpected manifest format: {type(data)}")
                    return False
            logger.info(f"Loaded {len(self.collection)} wallpapers from {manifest_path}")
            return True
        
        logger.warning(f"Collection file not found: {manifest_path}")
        return False
    
    # =========================================================================
    # Validation Checks
    # =========================================================================
    
    def check_required_fields(self) -> ValidationResult:
        """Check all wallpapers have required fields (with alias support)."""
        errors = []
        missing_counts = {}
        
        for wp in self.collection:
            wp_id = wp.get("id", "unknown")
            for field, aliases in self.REQUIRED_FIELDS:
                # Check if field or any alias is present and not None
                has_field = (
                    (field in wp and wp[field] is not None and wp[field] != "") or
                    any(alias in wp and wp[alias] is not None and wp[alias] != "" for alias in aliases)
                )
                if not has_field:
                    errors.append(f"{wp_id}: missing '{field}'")
                    missing_counts[field] = missing_counts.get(field, 0) + 1
        
        passed = len(errors) == 0
        
        return ValidationResult(
            check_name="Required Fields",
            passed=passed,
            message=f"All {len(self.collection)} wallpapers have required fields" if passed
                    else f"{len(errors)} missing fields found",
            errors=errors[:20],  # Limit to first 20
            stats={"missing_counts": missing_counts}
        )
    
    def check_embedding_dimensions(self) -> ValidationResult:
        """Verify embedding dimensions are correct."""
        errors = []
        stats = {model: {"correct": 0, "wrong": 0, "missing": 0} 
                 for model in self.EMBEDDING_DIMENSIONS}
        
        for wp in self.collection:
            wp_id = wp.get("id", "unknown")
            embeddings = wp.get("embeddings", {})
            
            for model, expected_dim in self.EMBEDDING_DIMENSIONS.items():
                if model not in embeddings or embeddings[model] is None:
                    stats[model]["missing"] += 1
                else:
                    actual_dim = len(embeddings[model])
                    if actual_dim != expected_dim:
                        errors.append(
                            f"{wp_id}: {model} has {actual_dim} dims, expected {expected_dim}"
                        )
                        stats[model]["wrong"] += 1
                    else:
                        stats[model]["correct"] += 1
        
        passed = len(errors) == 0
        
        return ValidationResult(
            check_name="Embedding Dimensions",
            passed=passed,
            message="All embeddings have correct dimensions" if passed
                    else f"{len(errors)} dimension mismatches",
            errors=errors[:20],
            stats=stats
        )
    
    def check_duplicate_ids(self) -> ValidationResult:
        """Detect duplicate wallpaper IDs."""
        id_counts: dict[str, int] = {}
        duplicates = []
        
        for wp in self.collection:
            wp_id = wp.get("id", "")
            id_counts[wp_id] = id_counts.get(wp_id, 0) + 1
        
        for wp_id, count in id_counts.items():
            if count > 1:
                duplicates.append(f"{wp_id}: appears {count} times")
        
        passed = len(duplicates) == 0
        
        return ValidationResult(
            check_name="Duplicate IDs",
            passed=passed,
            message=f"All {len(self.collection)} IDs are unique" if passed
                    else f"{len(duplicates)} duplicate IDs found",
            errors=duplicates,
            stats={"unique_ids": len(id_counts), "total_entries": len(self.collection)}
        )
    
    async def check_r2_urls(self, sample_size: int = 50) -> ValidationResult:
        """Spot-check R2 URLs are accessible via HEAD requests."""
        urls_to_check = []
        
        for wp in self.collection:
            url = wp.get("r2_url", "")
            if url:
                urls_to_check.append((wp.get("id", "unknown"), url))
        
        if not urls_to_check:
            return ValidationResult(
                check_name="R2 URL Accessibility",
                passed=True,
                message="No R2 URLs to check",
                stats={"checked": 0, "accessible": 0, "failed": 0}
            )
        
        # Random sample
        if len(urls_to_check) > sample_size:
            urls_to_check = random.sample(urls_to_check, sample_size)
        
        errors = []
        accessible = 0
        
        async with aiohttp.ClientSession() as session:
            for wp_id, url in urls_to_check:
                try:
                    async with session.head(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            accessible += 1
                        else:
                            errors.append(f"{wp_id}: HTTP {resp.status}")
                except Exception as e:
                    errors.append(f"{wp_id}: {str(e)[:50]}")
        
        failed = len(urls_to_check) - accessible
        passed = failed == 0 or (failed / len(urls_to_check)) < 0.1  # Allow 10% failure
        
        return ValidationResult(
            check_name="R2 URL Accessibility",
            passed=passed,
            message=f"{accessible}/{len(urls_to_check)} URLs accessible" if passed
                    else f"{failed} URLs failed (>{10}% threshold)",
            errors=errors,
            warnings=[] if passed else [f"{failed} URLs not accessible"],
            stats={
                "checked": len(urls_to_check),
                "accessible": accessible,
                "failed": failed,
                "sample_size": sample_size,
            }
        )
    
    def check_quality_scores(self) -> ValidationResult:
        """Validate quality scores are in expected range."""
        errors = []
        score_sum = 0
        score_count = 0
        below_threshold = 0
        
        # Get threshold from central config
        if HAS_CONFIG_LOADER:
            threshold = get_config().get('quality.threshold', 0.40)
        else:
            threshold = 0.40
        
        for wp in self.collection:
            wp_id = wp.get("id", "unknown")
            score = wp.get("quality_score")
            
            if score is None:
                errors.append(f"{wp_id}: missing quality_score")
            elif not isinstance(score, (int, float)):
                errors.append(f"{wp_id}: quality_score is not a number")
            elif score < 0 or score > 1:
                errors.append(f"{wp_id}: quality_score {score} out of range [0,1]")
            else:
                score_sum += score
                score_count += 1
                if score < threshold:
                    below_threshold += 1
        
        avg_score = score_sum / score_count if score_count > 0 else 0
        passed = len(errors) == 0
        
        return ValidationResult(
            check_name="Quality Scores",
            passed=passed,
            message=f"Average quality: {avg_score:.3f} (threshold: {threshold})" if passed
                    else f"{len(errors)} invalid quality scores",
            errors=errors[:20],
            warnings=[f"{below_threshold} wallpapers below {threshold} threshold"] if below_threshold else [],
            stats={
                "average_score": round(avg_score, 4),
                "threshold": threshold,
                "below_threshold": below_threshold,
                "valid_scores": score_count,
            }
        )
    
    def check_category_distribution(self) -> ValidationResult:
        """Generate category distribution summary (informational, always passes)."""
        categories: dict[str, int] = {}
        no_category = 0
        
        for wp in self.collection:
            category = wp.get("primary_category", "") or wp.get("category", "")
            if category:
                categories[category] = categories.get(category, 0) + 1
            else:
                no_category += 1
        
        total = len(self.collection)
        
        # Build summary info
        distribution_info = []
        for category, count in sorted(categories.items(), key=lambda x: -x[1]):
            percentage = count / total * 100 if total > 0 else 0
            distribution_info.append(f"{category}: {percentage:.1f}% ({count})")
        
        return ValidationResult(
            check_name="Category Distribution",
            passed=True,  # Always passes - just informational
            message=f"{len(categories)} categories found",
            errors=[],
            warnings=[f"{no_category} wallpapers have no category"] if no_category else [],
            stats={
                "distribution": {k: v for k, v in sorted(categories.items(), key=lambda x: -x[1])},
                "no_category": no_category,
                "summary": distribution_info,
            }
        )
    
    def check_phash_collisions(self) -> ValidationResult:
        """Detect images with identical perceptual hashes."""
        phash_map: dict[str, list[str]] = {}
        
        for wp in self.collection:
            wp_id = wp.get("id", "unknown")
            phash = wp.get("phash", "")
            
            if phash:
                if phash not in phash_map:
                    phash_map[phash] = []
                phash_map[phash].append(wp_id)
        
        collisions = []
        for phash, ids in phash_map.items():
            if len(ids) > 1:
                collisions.append(f"{phash}: {', '.join(ids)}")
        
        passed = len(collisions) == 0
        
        return ValidationResult(
            check_name="Perceptual Hash Collisions",
            passed=passed,
            message="No hash collisions" if passed
                    else f"{len(collisions)} potential duplicates found",
            errors=collisions[:20],
            stats={
                "unique_hashes": len(phash_map),
                "collision_count": len(collisions),
            }
        )
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def generate_statistics(self) -> dict:
        """Generate collection statistics."""
        if not self.collection:
            return {}
        
        # Date analysis
        dates = []
        for wp in self.collection:
            date_added = wp.get("date_added", "")
            if date_added:
                try:
                    dates.append(datetime.fromisoformat(date_added.replace("Z", "+00:00")))
                except:
                    pass
        
        if dates:
            dates.sort()
            oldest = dates[0].isoformat()
            newest = dates[-1].isoformat()
            days_span = (dates[-1] - dates[0]).days if len(dates) > 1 else 0
            avg_per_day = len(dates) / max(days_span, 1)
        else:
            oldest = newest = "N/A"
            avg_per_day = 0
        
        # Source analysis
        sources: dict[str, int] = {}
        for wp in self.collection:
            source = wp.get("source", "unknown")
            sources[source] = sources.get(source, 0) + 1
        
        # Resolution analysis
        resolutions: dict[str, int] = {}
        for wp in self.collection:
            width = wp.get("width", 0)
            height = wp.get("height", 0)
            if width and height:
                res = f"{width}x{height}"
                resolutions[res] = resolutions.get(res, 0) + 1
        
        return {
            "total_wallpapers": len(self.collection),
            "date_range": {"oldest": oldest, "newest": newest},
            "avg_per_day": round(avg_per_day, 2),
            "sources": sources,
            "top_resolutions": dict(sorted(resolutions.items(), 
                                           key=lambda x: -x[1])[:10]),
        }
    
    # =========================================================================
    # Main Runner
    # =========================================================================
    
    async def run_full_validation(self) -> ValidationReport:
        """Run all validation checks."""
        report = ValidationReport(
            timestamp=datetime.now().isoformat(),
            overall_passed=True,
        )
        
        if not self.load_collection():
            report.overall_passed = False
            report.checks.append(ValidationResult(
                check_name="Load Collection",
                passed=False,
                message="Failed to load collection file"
            ))
            return report
        
        report.total_wallpapers = len(self.collection)
        
        # Run checks
        checks = [
            self.check_required_fields(),
            self.check_embedding_dimensions(),
            self.check_duplicate_ids(),
            await self.check_r2_urls(sample_size=50),
            self.check_quality_scores(),
            self.check_category_distribution(),
            self.check_phash_collisions(),
        ]
        
        for check in checks:
            report.checks.append(check)
            if not check.passed:
                report.overall_passed = False
        
        report.summary = self.generate_statistics()
        
        return report
    
    async def run_quick_validation(self) -> ValidationReport:
        """Run quick validation (skip slow checks like URL verification)."""
        report = ValidationReport(
            timestamp=datetime.now().isoformat(),
            overall_passed=True,
        )
        
        if not self.load_collection():
            report.overall_passed = False
            report.checks.append(ValidationResult(
                check_name="Load Collection",
                passed=False,
                message="Failed to load collection file"
            ))
            return report
        
        report.total_wallpapers = len(self.collection)
        
        # Run quick checks only
        checks = [
            self.check_required_fields(),
            self.check_duplicate_ids(),
            self.check_quality_scores(),
            self.check_category_distribution(),
        ]
        
        for check in checks:
            report.checks.append(check)
            if not check.passed:
                report.overall_passed = False
        
        report.summary = self.generate_statistics()
        
        return report
    
    def save_report(self, report: ValidationReport, output_path: Path) -> None:
        """Save validation report to file."""
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Report saved to {output_path}")
    
    def print_report(self, report: ValidationReport) -> None:
        """Print formatted validation report."""
        print("\n" + "=" * 70)
        print("RAMEN COLLECTION VALIDATION REPORT")
        print("=" * 70)
        print(f"Timestamp: {report.timestamp}")
        print(f"Total Wallpapers: {report.total_wallpapers}")
        print(f"Overall Status: {'✅ PASSED' if report.overall_passed else '❌ FAILED'}")
        print("-" * 70)
        
        for check in report.checks:
            status = "✅" if check.passed else "❌"
            print(f"\n{status} {check.check_name}")
            print(f"   {check.message}")
            
            if check.errors:
                print(f"   Errors ({len(check.errors)}):")
                for error in check.errors[:5]:
                    print(f"     - {error}")
                if len(check.errors) > 5:
                    print(f"     ... and {len(check.errors) - 5} more")
            
            if check.warnings:
                print(f"   Warnings:")
                for warning in check.warnings:
                    print(f"     ⚠️  {warning}")
        
        if report.summary:
            print("\n" + "-" * 70)
            print("STATISTICS")
            print("-" * 70)
            print(json.dumps(report.summary, indent=2))
        
        print("\n" + "=" * 70)


async def main():
    parser = argparse.ArgumentParser(
        description="Validate RAMEN wallpaper collection health"
    )
    parser.add_argument(
        "--full-check",
        action="store_true",
        help="Run full validation including R2 URL checks"
    )
    parser.add_argument(
        "--quick-check",
        action="store_true",
        help="Run quick validation (skip slow checks)"
    )
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=Path("./manifests"),
        help="Path to manifests directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save report to JSON file"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output"
    )
    
    args = parser.parse_args()
    
    validator = CollectionValidator(manifests_dir=args.manifests_dir)
    
    if args.full_check:
        report = await validator.run_full_validation()
    else:
        report = await validator.run_quick_validation()
    
    if not args.quiet:
        validator.print_report(report)
    
    if args.output:
        validator.save_report(report, args.output)
    
    # Exit with appropriate code
    sys.exit(0 if report.overall_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
