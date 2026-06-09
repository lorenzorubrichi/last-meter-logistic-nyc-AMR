# GitHub Publication Checklist

Before publishing:

1. Remove all API keys and local credential files.
2. Do not commit `.venv`, `.idea`, cache folders, or raw downloaded images.
3. Confirm licenses for each NYC raw dataset.
4. Confirm whether derived Street View labels can be redistributed.
5. Keep only small sample data in GitHub.
6. Put large processed datasets in GitHub Releases, Zenodo, OSF, or Hugging Face Datasets.
7. Run `export_public_dataset.py` to rename ambiguous feature columns.
8. Add a release tag such as `v0.1.0`.
9. Add citation metadata once author names and repository URL are final.
10. Replace sample files in `data/raw/` with complete raw datasets before full reproduction.
11. Verify that raw-data source URLs in `docs/raw_data_sources.md` are final.

## Files Usually Included

- `README.md`
- `requirements.txt`
- `pyproject.toml`
- `.gitignore`
- `.env.example`
- `CITATION.cff`
- license file for code
- license note for data
- `docs/`
- `src/`
- `data/sample/`
- `data/raw/` with small schema-compatible extracts only
