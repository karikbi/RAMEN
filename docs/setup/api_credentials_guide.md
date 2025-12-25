# API Credentials & GitHub Secrets Setup Guide

This guide provides step-by-step instructions to generate the necessary API keys and add them to your GitHub repository secrets.

## 1. Reddit API Credentials

**Good news: You do NOT need any API credentials for Reddit.**

The script has been updated to use Reddit's public JSON feeds, which means:
*   No "Builder" approval needed.
*   No Client ID or Secret required.
*   No OAuth setup complexity.
*   Just works out of the box.

You can skip directly to setting up Unsplash and Pexels keys.

## 2. Unsplash API Key

1.  **Register**: Go to [Unsplash Developers](https://unsplash.com/developers) and "Register as a developer".
2.  **New Application**: Click **"New Application"**.
3.  **Accept Terms**: Check all the boxes and click "Accept terms".
4.  **App Info**:
    *   **Application Name**: `RAMEN Wallpapers`.
    *   **Description**: `Personal wallpaper curation script`.
5.  **Create**: Click **"Create application"**.
6.  **Copy Key**: Scroll down to the "Keys" section.
    *   Copy the **Access Key**. Save this as `UNSPLASH_ACCESS_KEY`.
    *   (You do not need the Secret Key for this script).

## 3. Pexels API Key

1.  **Request Key**: Go to [Pexels API](https://www.pexels.com/api/new/).
2.  **Form**:
    *   **Reason**: Select "Personal project".
    *   **Description**: `Wallpaper quality analysis and curation`.
    *   **Site URL**: You can put your GitHub repo URL or leave blank if optional.
3.  **Generate**: Click "Generate API Key".
4.  **Copy**: Copy the long string provided. Save this as `PEXELS_API_KEY`.

## 4. Cloudflare R2 (Optional/Advanced)

If you are setting up R2 storage:
1.  **Dashboard**: Go to Cloudflare Dashboard > **R2**.
2.  **Create Bucket**: Create a bucket named `ramen-wallpapers` (or similar). Save this name as `R2_BUCKET_NAME`.
3.  **API Tokens**: On the right, click **"Manage R2 API Tokens"**.
4.  **Create Token**:
    *   **Permissions**: Select **Admin Read & Write**.
    *   **TTL**: "Forever" or as desired.
5.  **Copy Details**:
    *   **Access Key ID**: Save as `R2_ACCESS_KEY`.
    *   **Secret Access Key**: Save as `R2_SECRET_KEY`.
    *   **Endpoint**: Copy the URL for your bucket (usually `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`). Save as `R2_ENDPOINT`.

---

## 5. Adding Secrets to GitHub

Since you cannot use the CLI, use the browser:

1.  **Navigate**: Go to your GitHub repository page (e.g., `https://github.com/avinash/RAMEN`).
2.  **Settings**: Click the **Settings** tab (gear icon) at the top right.
3.  **Security**: On the left sidebar, scroll down to **Secrets and variables**.
4.  **Actions**: Click on **Actions** (under Secrets and variables).
5.  **Add Secret**: Click the green **New repository secret** button.
6.  **Enter Data**:
    *   **Name**: Paste the name (e.g., `REDDIT_CLIENT_ID`).
    *   **Secret**: Paste the value you copied earlier.
    *   Click **Add secret**.
7.  **Repeat**: Repeat step 6 for all the keys you collected:
    *   `REDDIT_CLIENT_ID`
    *   `REDDIT_CLIENT_SECRET`
    *   `REDDIT_USER_AGENT`
    *   `UNSPLASH_ACCESS_KEY`
    *   `PEXELS_API_KEY`
    *   (And R2 secrets if you have them)
