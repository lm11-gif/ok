# Import the ORBIT Seed model binaries

The repository includes a GitHub Actions workflow at
`.github/workflows/import-orbit-model.yml`. It copies the two binary model
files from a GitHub Release into `orbit/seed_model/`, then commits them to
`main`.

## One-time upload

1. Open https://github.com/lm11-gif/ok/releases/new while signed in.
2. Create a tag such as `orbit-seed-upload` and use a release title such as
   `ORBIT Seed model upload`.
3. Attach the original archive named exactly
   `orbit-seed-gpu-ready.zip`. Do not rename it.
4. Publish the release.

Publishing the release starts the workflow automatically. It finds
`model.safetensors` and `tokenizer.model` inside the ZIP, checks that both
are present and that the safetensors header is readable, then commits those
two files to `main`.

The workflow uses the repository's built-in `GITHUB_TOKEN`; no personal
access token is needed. Check the **Actions** tab for the workflow run and
its result. The release asset itself remains attached to the release.

## Manual rerun

If the release event did not trigger, open **Actions → Import ORBIT Seed
model binaries → Run workflow**, enter the release tag, and run it. The
workflow can be rerun; if the two files are already identical, it makes no
additional commit.
