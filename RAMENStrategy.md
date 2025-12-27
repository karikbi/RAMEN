# Final Strategy: Premium Wallpaper Collection

## Mission
Build a 10,000-15,000 exceptional wallpaper collection with 4-model embeddings and rich metadata, fully automated via GitHub Actions, stored on Cloudflare R2.

***

## Core Principles

**Quality over quantity**: No fixed quotas. Accept only wallpapers scoring ≥5.5 on Aesthetic V2.5 (1-10 scale)

**Automated curation**: GitHub Actions runs daily at 2 AM IST, zero manual work except weekly reviews

**Four embedding models** (V2 stack): Capture visual similarity (EfficientNetV2), semantic understanding (SigLIP 2), composition (DINOv3), and device compatibility (MobileNetV4)

**Premium metadata**: 20+ fields per wallpaper including colors, categories, scene analysis, composition metrics, artist attribution

**Sustainable costs**: Under $20/year total infrastructure

***

## The 4-Model System (V2 Stack)

**MobileNetV4-Small** (1280 dimensions - timm includes 960→1280 projection layer)
- Bridge between user uploads and better models
- Enables Personalize Mode in Vanderwaals
- 2× faster, +7% accuracy vs MobileNetV3

**EfficientNetV2-Large** (1,280 dimensions)
- Best visual similarity and aesthetic quality
- Primary recommendation engine
- Captures artistic style, lighting, composition

**SigLIP 2 Large** (1,152 dimensions)
- Semantic understanding and concepts
- Auto-categorization and tagging
- Better localization, multilingual support

**DINOv3-Large** (1,024 dimensions)
- Scene composition and spatial relationships
- Artistic principles (rule of thirds, symmetry, balance)
- +6 mIoU composition improvement

**Storage**: ~6.2KB per wallpaper (all 4 embeddings + metadata), ~47MB for 10,000 wallpapers compressed

***

## Data Sources

**Reddit (60%)**: r/wallpapers, r/EarthPorn, r/Amoledbackgrounds, r/MinimalWallpaper, r/CityPorn
- Top monthly posts, 1,000-10,000+ upvotes depending on subreddit size
- Fetch 100-150 candidates daily

**Stock Photography (25%)**: Unsplash, Pexels, Pixabay
- Curated collections and featured sections only (not generic search)
- Fetch 30-50 candidates daily

**Community & Manual (15%)**: Behance, Dribbble, GitHub Issues submissions
- Weekly manual curation (1 hour)
- Accept ~20% of community submissions

**Total**: 150-200 candidates daily initially, scale to 300-500 as sources proven

***

## Quality Standards (No Fixed Quotas)

### Hard Requirements (Auto-Reject)
- Resolution: Minimum 2560×1440 (QHD)
- Uniqueness: <85% perceptual hash similarity to existing
- Text: <30% coverage (reject screenshots)
- File quality: No severe compression artifacts

### Quality Score (Accept if ≥ Threshold)

**Aesthetic Predictor V2.5** (Primary - 1-10 scale):
- 6.5 - 10: Premium (feature in "Best")
- 5.5 - 6.5: Standard (general catalog)
- 4.0 - 5.5: Acceptable (lower priority)
- < 4.0: Low (filter out)

**Plus SigLIP checks** for technical quality and wallpaper suitability.

**Thresholds adjust by collection phase**:
- Early (0-1,000): 5.0 (build diverse foundation)
- Growth (1,000-5,000): 5.5 (standard quality)
- Mature (5,000-10,000): 6.0 (selective)
- Rotation (10,000+): 6.5 (only exceptional additions)

**Daily acceptance varies naturally**: Some days 3 wallpapers pass, other days 40 pass. Quality is the only gatekeeper.

***

## Daily Automated Pipeline

### GitHub Actions (30-40 minutes daily at 2 AM IST)

**Stage 1: Fetch** (5-8 min)
- Connect to Reddit, Unsplash, Pexels APIs
- Download 150-200 candidate images
- Store original URLs for attribution

**Stage 2: Filter** (8-12 min)
- Validate resolution, format, file integrity
- Perceptual hash deduplication vs existing collection
- OCR text detection, NSFW filtering
- Preliminary quality scoring
- Result: ~10-20 candidates pass to deep processing

**Stage 3: Process** (15-20 min)
- Extract embeddings from all 4 models (~5 seconds per wallpaper)
- Generate metadata: color palettes, categories, scene tags
- Composition analysis and final quality scoring
- Result: Wallpapers scoring ≥5.5/10 (Aesthetic V2.5) approved for upload

**Stage 4: Upload** (3-5 min)
- Batch upload approved wallpapers to Cloudflare R2 via rclone
- Update manifest JSON with new entries
- Commit manifest to Vanderwaals repository

**Stage 5: Report** (2 min)
- Generate statistics (candidates, accepted, rejected breakdown)
- Create GitHub Issue if failures detected
- Track quality trends and source performance

**Output**: Variable daily (3-40 wallpapers typical, average 12-18), all exceeding quality threshold

***

## Storage Architecture

### Cloudflare R2 (Images)
- Original quality files (5MB average per wallpaper)
- Organization: `/2025/01/category/wp_id.ext`
- Batch uploads for cost efficiency
- On-demand compression via Cloudflare Workers (no pre-stored sizes)

**Cost**:
- 5,000 wallpapers: $0.38/month
- 10,000 wallpapers: $0.75/month
- 15,000 wallpapers: $1.13/month

### GitHub Repository (Metadata)
- Manifest JSON with all metadata + embeddings
- 44MB for 10,000 wallpapers (compressed)
- Served via jsDelivr CDN (free)
- Delta files for weekly updates (~500KB)

**Cost**: $0

**Total infrastructure**: Under $20/year

***

## Smart Source Optimization

### Adaptive Balancing (No Manual Tuning)

**Track acceptance rate per source** (30-day rolling):
- High performers (30%+ pass rate): Increase allocation
- Medium performers (10-20%): Maintain allocation
- Low performers (<5%): Reduce or remove

**Example adjustment**:
- r/EarthPorn: 35% pass rate → Increase from 30 to 60 candidates
- Pexels: 6% pass rate → Reduce from 20 to 5 candidates

**Sources optimize automatically** based on quality delivered, not manual guessing.

***

## Growth Timeline

### Phase 1: Foundation (Months 1-3)
- Target: 1,500-2,000 wallpapers
- Threshold: 5.0/10 (build diverse base)
- Focus: Validate pipeline, tune filters
- Typical: 10-20 accepted daily

### Phase 2: Expansion (Months 4-6)
- Target: 5,000 wallpapers total
- Threshold: 5.5/10 (standard quality)
- Focus: Add specialized sources, scale up candidates
- Typical: 15-25 accepted daily

### Phase 3: Maturity (Months 7-12)
- Target: 8,000-10,000 wallpapers
- Threshold: 6.0/10 (selective)
- Focus: Quarterly rotation, text search feature
- Typical: 12-18 accepted daily

### Phase 4: Living Collection (Year 2+)
- Target: 10,000-15,000 maintained
- Threshold: 6.5/10 (exceptional only)
- Focus: Replace low-performers, community-driven
- Add new while removing old (net stable size)

**Timeline to 10,000**: 12-24 months depending on source quality
**No rush**: Quality determines pace, not arbitrary deadlines

***

## Premium Metadata (Per Wallpaper)

### Core Identification
- Unique ID, title, date added, version

### Visual Characteristics
- 5-color palette (LAB color space)
- Dominant hue, brightness, contrast ratio
- Color diversity score

### Scene Understanding (AI-Generated)
- Primary category and subcategories
- Scene elements (water, sky, mountains, etc.)
- Time of day, weather, season
- Subject matter

### Composition Analysis
- Composition type (rule of thirds, symmetry, etc.)
- Symmetry score, depth perception
- Complexity level, focal point coordinates

### Aesthetic Properties
- Mood tags (calm, energetic, dramatic, etc.)
- Style tags (Gruvbox, Nord, cyberpunk, etc.)
- Quality tier (Premium/Standard)
- Aesthetic score

### Technical Metadata
- Dimensions, file size, aspect ratio, format
- EXIF data (when available)
- Storage URLs

### Attribution & Licensing
- Original source URL and platform
- Artist name and profile
- License type, copyright string
- Reddit-specific: subreddit, upvotes, post URL

### Embeddings
- All 4 models stored as binary BLOBs (quantized)

***

## Integration with Vanderwaals

### Personalize Mode (Bridge Strategy)
1. User uploads favorite wallpaper
2. On-device MobileNetV4 extracts embedding (faster than MobileNetV3)
3. Find top 30 matches using MobileNetV4 embeddings (or 576D legacy projection)
4. User confirms 4-10 they actually like
5. Average their EfficientNetV2 embeddings = user preference
6. Future recommendations use EfficientNetV2 (superior quality)

### Auto Mode (Direct Learning)
1. User starts without upload
2. First like initializes EfficientNetV2 preference
3. All learning in superior embedding space
4. MobileNetV4 bypassed entirely

### Text Search (Future)
1. User types "sunset beach"
2. Cloudflare Worker encodes via SigLIP 2 (~20ms)
3. Compare against local SigLIP 2 embeddings
4. Instant semantic search results

***

## Success Metrics

### Collection Health (Weekly Monitoring)
- Average quality score (should stay ≥6.0/10 on V2.5)
- Category balance (no category >35% or <5%)
- Source acceptance rates (optimize allocation)
- Diversity score (embeddings spread across space)

### Technical Performance
- GitHub Actions success rate (target: 95%+)
- Average processing time (~30-40 min)
- R2 upload success rate (target: 100%)
- Metadata completeness (zero missing fields)

### Weekly Manual Reviews
- Spot-check 20 random accepted wallpapers (personal satisfaction)
- Review borderline queue (0.80-0.84 scores)
- Address any quality alerts from automation

***

## The Bottom Line

**What you're building**: World-class wallpaper dataset with best-in-class embeddings and richest metadata

**How it works**: Fully automated GitHub Actions pipeline, quality-first approach (no quotas)

**Timeline**: 12-24 months to 10,000 exceptional wallpapers (pace determined by quality, not targets)

**Cost**: Under $20/year total infrastructure

**Effort**: 30 min daily automated + 1 hour weekly manual review

**Result**: When integrated into Vanderwaals, users get recommendations better than any wallpaper app because your foundation is bulletproof. Every wallpaper is genuinely exceptional. No filler, no compromises.

**Quality is the strategy. Everything else is just execution.**

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/56197477/1b90ee80-0b07-4676-8bdf-601fc51d7fec/README.md)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/56197477/b75be730-91c8-4fe3-b143-57d7a8835210/CHANGELOG.md)