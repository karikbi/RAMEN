# Manifests Directory

> **Note**: Manifests are now stored on Cloudflare R2 for security.
> This directory contains local cache files only.

## Storage Location

Manifests are stored on Cloudflare R2:
- **Collection manifest**: `manifests/collection.json.gz`
- **Delta files**: `manifests/deltas/delta_YYYY_MM_DD.json.gz`
- **Dedup index**: `dedup/index.json.gz`

### Why R2 Instead of GitHub?

1. **Security**: Keep collection data private (not exposed in public repo)
2. **Size**: Avoid bloating the GitHub repository with large manifest files
3. **Performance**: R2 CDN provides faster access for clients
4. **Cost**: R2 free tier is generous for our use case

## Local Cache Structure

```
manifests/
└── README.md           # This file

manifest_cache/
├── collection.json.gz  # Local cache of latest manifest

dedup_cache/
└── index.json.gz       # Local cache of dedup index
```

## Manifest Format

Each manifest is a compressed JSON file containing an array of wallpaper objects:

```json
{
  "version": "1.0",
  "updated": "2025-12-25T02:00:00Z",
  "count": 1234,
  "wallpapers": [
    {
      "id": "wp_abc123",
      "title": "Mountain Sunset",
      "url": "https://r2.example.com/wallpapers/wp_abc123.jpg",
      "category": "nature",
      "subcategories": ["mountain", "sunset"],
      "colors": ["#FF5733", "#33FF57", "#3357FF"],
      "dominant_hue": 45,
      "brightness": 0.65,
      "quality_score": 0.89,
      "quality_tier": "premium",
      "dimensions": { "width": 3840, "height": 2160 },
      "aspect_ratio": 1.78,
      "date_added": "2025-12-25T02:00:00Z",
      "source": "reddit",
      "artist": "photographer_name",
      "embeddings": {
        "mobilenet_v3": "base64_encoded_vector",
        "efficientnet_v2": "base64_encoded_vector",
        "siglip": "base64_encoded_vector",
        "dinov2": "base64_encoded_vector"
      }
    }
  ]
}
```

## Access

### For Pipeline (Environment Variables)

The pipeline requires R2 credentials to access manifests:

```bash
R2_ENDPOINT=https://your_account_id.r2.cloudflarestorage.com
R2_ACCESS_KEY=your_access_key
R2_SECRET_KEY=your_secret_key
R2_BUCKET_NAME=your_bucket_name
```

### For Client Apps

Contact the maintainer for access details. Options include:
- Making the R2 bucket public
- Using signed URLs with expiration
- Creating a read-only API key

## Deduplication Index

The dedup index is stored on R2 at `dedup/index.json.gz` and contains:

- **Seen URLs**: Source URLs already fetched
- **Seen IDs**: Wallpaper IDs already processed
- **Perceptual Hashes**: For visual duplicate detection
- **Content Hashes**: SHA256 for exact binary match

This enables duplicate prevention across pipeline runs, even when running on ephemeral CI environments like GitHub Actions.

## Notes

- Manifests are automatically generated and synced by the pipeline
- Delta files contain only new additions since the last full manifest
- The dedup index is synced before and after each pipeline run
- Local caches are used for faster access and offline fallback
