# src/instagram_scraper.py

import instaloader
import requests
from PIL import Image
from io import BytesIO
import re
import tempfile
import os


def extract_username(url_or_username: str) -> str:
    """
    Accepts either a full Instagram URL or a plain username and returns the username.
    Handles URLs like:
      - https://www.instagram.com/lakshayisnt/
      - https://www.instagram.com/lakshayisnt?igsh=abc123
    """
    url_or_username = url_or_username.strip()
    match = re.search(r"instagram\.com/([A-Za-z0-9_.]+)", url_or_username)
    if match:
        return match.group(1)
    return url_or_username  # assume it's already a plain username


def scrape_instagram_profile(url_or_username: str, ig_username: str = None, ig_password: str = None) -> dict:
    """
    Scrapes a public Instagram profile and returns:
    {
        "image": PIL.Image,
        "username": str,
        "bio": str,
        "followers": int,
        "following": int,
        "posts": int,
        "is_private": bool,
        "is_verified": bool,
        "error": str or None
    }

    ig_username / ig_password are optional. Without login,
    instaloader still works for most public profiles but may
    get rate-limited faster.
    """
    username = extract_username(url_or_username)
    result = {
        "image": None,
        "username": username,
        "bio": "",
        "followers": 0,
        "following": 0,
        "posts": 0,
        "is_private": False,
        "is_verified": False,
        "error": None,
    }

    try:
        L = instaloader.Instaloader(
            download_pictures=False,
            download_videos=False,
            download_video_thumbnails=False,
            download_geotags=False,
            download_comments=False,
            save_metadata=False,
            quiet=True,
        )

        # Optional login for higher rate limits
        if ig_username and ig_password:
            try:
                L.login(ig_username, ig_password)
            except Exception as login_err:
                result["error"] = f"Login failed (continuing anonymously): {login_err}"

        profile = instaloader.Profile.from_username(L.context, username)

        result["bio"] = profile.biography or ""
        result["followers"] = profile.followers
        result["following"] = profile.followees
        result["posts"] = profile.mediacount
        result["is_private"] = profile.is_private
        result["is_verified"] = profile.is_verified

        # Download profile picture
        pic_url = profile.profile_pic_url
        response = requests.get(pic_url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        result["image"] = image

    except instaloader.exceptions.ProfileNotExistsException:
        result["error"] = f"Profile '@{username}' does not exist."
    except instaloader.exceptions.PrivateProfileNotFollowedException:
        result["error"] = f"Profile '@{username}' is private."
    except instaloader.exceptions.ConnectionException as e:
        result["error"] = f"Instagram connection error (rate limit?): {e}"
    except Exception as e:
        result["error"] = f"Unexpected error: {e}"

    return result