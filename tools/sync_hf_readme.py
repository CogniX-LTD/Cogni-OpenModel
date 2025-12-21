import os
import sys
from huggingface_hub import HfApi

def main():
    repo_id = os.environ.get("HF_REPO_ID")
    if not repo_id and len(sys.argv) > 1:
        repo_id = sys.argv[1]
    token = os.environ.get("HF_TOKEN")
    if not repo_id:
        raise ValueError("HF_REPO_ID environment variable or first argument is required")
    if not token:
        raise ValueError("HF_TOKEN environment variable is required")
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    card_path = os.path.join(base_dir, "README.hf.md")
    if not os.path.exists(card_path):
        raise FileNotFoundError("README.hf.md not found")
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=card_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Sync README.hf.md to README.md for proper model card metadata",
    )
    print("Uploaded README.hf.md to", repo_id)

if __name__ == "__main__":
    main()
