import os
import sys
from huggingface_hub import HfApi

def sync_readme(repo_id=None):
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable not set.")
        return

    repo_id = repo_id or os.getenv("HF_REPO_ID")
    if not repo_id:
        print("Error: HF_REPO_ID environment variable not set and no repo_id provided as argument.")
        print("Usage: python tools/sync_hf_readme.py <org/model>")
        return

    api = HfApi()
    
    local_path = "README.hf.md"
    if not os.path.exists(local_path):
        print(f"Error: {local_path} not found.")
        return

    print(f"Uploading {local_path} to {repo_id} as README.md...")
    try:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            token=hf_token
        )
        print("Successfully synced README to Hugging Face!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    target_repo = sys.argv[1] if len(sys.argv) > 1 else None
    sync_readme(target_repo)
