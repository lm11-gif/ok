# Import the ORBIT Seed model binaries

The GitHub Actions workflow at
`.github/workflows/import-orbit-model.yml` copies the two binary model
files from a GitHub Release into `orbit/seed_model/`, validates them, and
commits them to `main`.

## Your current release

The release tag is `orbit-seed-upload`. It contains the two separate assets:
- `model.safetensors`
- `tokenizer.model`

The importer has been updated to accept these two separate assets. The first
release-triggered run was skipped because the earlier workflow mistakenly
looked for a ZIP archive. The corrected workflow needs to be started manually
for this already-published release.

## Start the import

1. Open https://github.com/lm11-gif/ok/actions/workflows/import-orbit-model.yml
2. Select **Run workflow**.
3. Keep branch set to `main`.
4. Enter `orbit-seed-upload` as the release tag.
5. Click **Run workflow**.

The workflow downloads the two files from the release using the repository's
built-in `GITHUB_TOKEN`, checks that both files exist and that the
safetensors header is readable, then commits them to `main`.

No personal access token is needed. Check the Actions run for the result.
The release assets remain attached to the release.
