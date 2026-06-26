from huggingface_hub import HfApi, create_repo

username = "Theara2"
repo_name = "cambodia-news-classifier"

# Create repository
try:
    create_repo(repo_id=f"{username}/{repo_name}", repo_type="model")
    print(f"✅ Repository created: {username}/{repo_name}")
except Exception as e:
    print(f"ℹ️ Repository may already exist: {e}")

# Upload the model
api = HfApi()
api.upload_file(
    path_or_fileobj="roberta_best.pt",
    path_in_repo="roberta_best.pt",
    repo_id=f"{username}/{repo_name}",
    repo_type="model"
)

print(f"✅ Model uploaded to: https://huggingface.co/{username}/{repo_name}")