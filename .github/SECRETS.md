# RAMEN Pipeline - GitHub Secrets Configuration

This document describes all the secrets required for the GitHub Actions workflow.

## Required Secrets

Configure these in your GitHub repository:
**Settings → Secrets and variables → Actions → New repository secret**

---

### Reddit API Credentials
 
| Secret | Description | How to Get |
|--------|-------------|------------|
| `REDDIT_USER_AGENT` | Descriptive User-Agent string | Use: `python:RAMEN-Wallpaper-Curator:v1.0.0 (by /u/wallpaper_curator)` |

**Note:** Reddit no longer requires OAuth credentials. The pipeline uses public JSON feeds.

---

### Stock Photo APIs

| Secret | Description | How to Get |
|--------|-------------|------------|
| `UNSPLASH_ACCESS_KEY` | Unsplash API access key | [Unsplash Developers](https://unsplash.com/developers) |
| `PEXELS_API_KEY` | Pexels API key | [Pexels API](https://www.pexels.com/api/) |

---

### Cloudflare R2 Storage

| Secret | Description | How to Get |
|--------|-------------|------------|
| `R2_ENDPOINT` | R2 bucket endpoint URL | Cloudflare Dashboard → R2 → Bucket → Settings |
| `R2_ACCESS_KEY` | R2 access key ID | Cloudflare Dashboard → R2 → Manage R2 API Tokens |
| `R2_SECRET_KEY` | R2 secret access key | Same as above |
| `R2_BUCKET_NAME` | Name of your R2 bucket | The bucket name you created |

**Example endpoint:** `https://<account_id>.r2.cloudflarestorage.com`

---

### GitHub Tokens

| Secret | Description | How to Get |
|--------|-------------|------------|
| `PERSONAL_GITHUB_TOKEN` | PAT with repo write access | [GitHub Settings → Developer Settings → Personal Access Tokens](https://github.com/settings/tokens) |

**Required scopes for PAT:**
- `repo` (Full control of private repositories)
- `workflow` (Update GitHub Action workflows)

> **Note:** `GITHUB_TOKEN` is automatically provided by GitHub Actions and doesn't need to be configured manually.

---

## Verification Checklist

After adding all secrets, verify your setup:

- [ ] All 8 secrets are configured
- [ ] No trailing whitespace in secret values
- [ ] `REDDIT_USER_AGENT` follows the format: `python:AppName:Version (by /u/username)`
- [ ] Unsplash/Pexels API keys are valid
- [ ] R2 bucket exists and is accessible
- [ ] PAT has correct scopes

## Testing

Run the workflow manually with dry-run mode to verify configuration:

1. Go to **Actions** → **Daily Wallpaper Curation**
2. Click **Run workflow**
3. Set `dry_run` to `true`
4. Click **Run workflow**

Check the logs for any authentication or configuration errors.

---

## Troubleshooting

### "Reddit API 403 Forbidden"
- Verify `REDDIT_USER_AGENT` is set and follows Reddit's format guidelines
- Format: `platform:app_name:version (by /u/reddit_username)`
- Avoid generic User-Agents like "bot" or "script"

### "Unsplash/Pexels rate limit exceeded"
- These APIs have daily/monthly limits
- Consider reducing fetch counts in `config.yaml`

### "R2 upload failed"
- Verify endpoint URL format
- Check bucket permissions (needs write access)
- Ensure bucket region matches endpoint

### "Push to main failed"
- Verify `PERSONAL_GITHUB_TOKEN` has `repo` scope
- Check if branch protection rules require different auth

