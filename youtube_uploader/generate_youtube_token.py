"""
generate_youtube_token.py

Purpose:
- Open Google sign-in in a browser.
- Request the required YouTube permissions.
- Create .credentials/youtube_token.json.
- The token file is used by run_upload.py / uploader.py to upload videos to YouTube.

Run this file only when creating a token for the first time,
when the existing token is invalid, or when signing in to a different YouTube account.
"""

import os

# This library runs the OAuth flow for a desktop/local application.
# Running it opens a browser for Google sign-in.
from google_auth_oauthlib.flow import InstalledAppFlow


# Scopes are the permissions requested for the Google/YouTube account.
# youtube.upload  -> permission to upload videos to YouTube.
# youtube.readonly -> permission to read channel/video data, such as existing upload schedules.
YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]


# OAuth Client ID file from Google Cloud Console.
# Find it at:
# Google Cloud → APIs & Services → Credentials → OAuth client ID → Desktop app → Download JSON
#
# Rename the downloaded file to client_secret.json
# and save it in the .credentials/ folder.
CLIENT_SECRET_FILE = ".credentials/client_secret.json"


# Output file for the Google sign-in token.
# uploader.py reads this file later.
TOKEN_FILE = ".credentials/youtube_token.json"


def main():
    """
    Generate a YouTube token.

    Steps:
    1. Check whether client_secret.json exists.
    2. Create the .credentials folder if needed.
    3. Open OAuth sign-in in a browser.
    4. After approval, save the token to youtube_token.json.
    """

    # Check whether client_secret.json is available.
    # Stop if it is missing because the OAuth flow cannot start.
    if not os.path.exists(CLIENT_SECRET_FILE):
        raise FileNotFoundError(
            f"{CLIENT_SECRET_FILE} was not found. "
            "Download the OAuth Client ID JSON from Google Cloud, rename it to client_secret.json, "
            "and save it in the .credentials/ folder."
        )

    # Ensure the .credentials folder exists; create it if needed.
    os.makedirs(".credentials", exist_ok=True)

    # Create the OAuth flow from client_secret.json with the required YouTube scopes.
    flow = InstalledAppFlow.from_client_secrets_file(
        CLIENT_SECRET_FILE,
        scopes=YOUTUBE_SCOPES,
    )

    # Run OAuth sign-in through a local browser.
    #
    # port=0:
    #   Python automatically selects an available port.
    #
    # access_type="offline":
    #   Request a refresh_token so the token can be renewed automatically
    #   without signing in again whenever the access_token expires.
    #
    # prompt="consent":
    #   Ask Google to show the consent screen again.
    #   This helps ensure that a refresh_token is issued.
    creds = flow.run_local_server(
        port=0,
        access_type="offline",
        prompt="consent",
    )

    # Save the credentials/token to youtube_token.json.
    # This file contains access_token, refresh_token, client_id, client_secret, and scopes.
    #
    # PENTING:
    # Do not upload youtube_token.json to GitHub.
    # Treat this file like your YouTube account password.
    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        f.write(creds.to_json())

    print(f"✅ Token created successfully: {TOKEN_FILE}")


# Run main() only when this file is executed directly:
#
# python generate_youtube_token.py
#
# Importing this file from another Python file does not run main().
if __name__ == "__main__":
    main()
