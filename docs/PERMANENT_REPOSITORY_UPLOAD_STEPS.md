# Permanent public repository upload steps

## Recommended release

Use the conservative package below as the first public release:

- Folder: `release/zenodo_public_20260914/`
- ZIP: `release/TimesFM_zenodo_public_20260914.zip`
- Version: `1.1.0`
- File count: 145 files in the folder; 145 files in the ZIP manifest plus the ZIP directory entries
- SHA-256 manifest: `release/zenodo_public_20260914/metadata/FILE_MANIFEST_SHA256.csv`

The package excludes raw monitoring exports and source arrays that contain observed water-level values. Do not add those files unless written permission from the Heilongjiang Earthquake Agency has been obtained.

## Route A: GitHub repository plus Zenodo DOI

1. Create a new public GitHub repository, for example `TimesFM-Beilin-groundwater`.
2. Copy the contents of `release/zenodo_public_20260914/` into the repository root. Do not copy the parent `release` directory.
3. In `CITATION.cff`, replace `https://github.com/REPLACE_WITH_REPOSITORY/TimesFM` with the actual repository URL.
4. Commit the files with a message such as `Release v1.1.0 supporting data and code`.
5. Create an annotated Git tag `v1.1.0` and push the tag.
6. Open the GitHub repository’s **Releases** page and create release `v1.1.0`. Upload `TimesFM_zenodo_public_20260914.zip` as an additional release asset if desired; the repository files themselves are the primary source.
7. Sign in to Zenodo, open **GitHub integration**, and enable the repository.
8. On Zenodo, open the newly created GitHub release and select **Edit** before publishing.
9. Copy the fields from `metadata/zenodo_metadata.json`: title, creators, description, keywords, version, and publication date.
10. Set the upload type to **Software**. If Zenodo requires a single archive-level licence, choose **Other** and write: “Code is MIT licensed; catalogue and observation-derived materials remain subject to the terms described in DATA_NOTICE.md.”
11. Add the GitHub repository URL as the related identifier. After the article is accepted or a DOI is assigned, add the article DOI as a related publication.
12. Publish the Zenodo record. Zenodo will issue a version DOI and a concept DOI. Use the concept DOI for the article’s long-term Data Availability Statement and the version DOI for exact-release citation.
13. Download the published Zenodo ZIP and compare its SHA-256 checksum with the local ZIP before citing it.

## Route B: Direct Zenodo upload

1. Sign in to Zenodo and choose **New upload**.
2. Upload `release/TimesFM_zenodo_public_20260914.zip`.
3. Use the same metadata and licence wording described in Route A.
4. In the description, state that raw monitoring exports are withheld under the data owner’s rules and that the package contains code, public catalogues, model predictions, anomaly segments, statistics, figures, tables, and permitted source arrays.
5. Add the manuscript title and manuscript identifier `applsci-4544483` as related information if the form provides that field.
6. Publish the record and verify the generated DOI, file download, README, metadata JSON, and SHA-256 manifest.

## Checks before publishing

Run these commands from the project root:

```powershell
.venv_revision\\Scripts\\python.exe scripts\\09_validate.py
.venv_revision\\Scripts\\python.exe scripts\\check_submission_outputs.py
.venv_revision\\Scripts\\python.exe scripts\\build_zenodo_public_package_20260914.py
Get-FileHash release\\TimesFM_zenodo_public_20260914.zip -Algorithm SHA256
```

Then confirm that the public package contains no reviewer reports, original submission, raw EQT exports, virtual environment, model checkpoint, or placeholder repository URL:

```powershell
rg -n -- "review_materials|original_submission|data/raw/.*TXT|model.safetensors|REPLACE_WITH_REPOSITORY" release\\zenodo_public_20260914
```

An empty result is required, except for the intentional `data/raw/README_RESTRICTED.txt` and the placeholder warning in `docs/DOI_METADATA_NOTES.md` before the URL is replaced.

## Updating the manuscript after DOI reservation

Replace the current placeholder sentence in the manuscript Data Availability Statement with the actual DOI. A suitable form is:

> The analysis code and non-sensitive derived results are available at Zenodo, https://doi.org/DOI_TO_BE_INSERTED. Raw hourly monitoring data are subject to restrictions imposed by the Heilongjiang Earthquake Agency and are available from the corresponding author with the permission of the data owner.

Update the same DOI in the response letter, `CITATION.cff`, the GitHub release description, and the Zenodo related-publication field. Keep the raw-data restriction statement unchanged unless the data owner grants broader permission.

## Optional second release after data-owner approval

If written permission later covers the corrected daily/hourly series, issue a new version (for example `1.2.0`) rather than replacing version 1.1.0. Add only the approved files, record the permission and licence in `DATA_NOTICE.md`, update the manifest, rerun all validation checks, and cite the new version DOI separately.
