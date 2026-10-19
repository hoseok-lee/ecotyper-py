# Installation

## Requirements

- Python ≥ 3.10
- numpy, scipy, pandas, anndata, numba, pyyaml (installed automatically)
- optional: **PyICU**, to sort strings the way R does in a UTF-8 locale (recommended for exact
  parity); **matplotlib** for your own plots
- a C compiler (`cc`) on x86-64, for the long-double kernels that make results bit-identical to R
  (built automatically on first use; see [Validation](validation.md))

The package is not on PyPI. Install it from a clone of the repository.

## With conda (recommended)

```bash
git clone https://github.com/hoseok-lee/ecotyper-py.git
cd ecotyper-py
conda create -n ecotyper-py -c conda-forge python=3.12 numpy scipy pandas anndata numba pyicu pyyaml
conda activate ecotyper-py
pip install -e .
```

## With pip

```bash
pip install -e ".[icu,plot]"      # PyICU needs the ICU C library (e.g. brew install icu4c)
```

Optional extras: `icu` (PyICU), `plot` (matplotlib), `test` (pytest), `docs` (this
documentation).

## With Docker

`docker/Dockerfile` builds an image with the dependencies at the exact versions used for the
validation. The package itself is mounted and installed at run time (as `work/dock.sh` does):

```bash
docker build -t ecotyper-py:dev -f docker/Dockerfile .
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$PWD:/work" -w /work ecotyper-py:dev \
  bash -c 'pip install -q --no-deps -e . --target /tmp/pkgs && PYTHONPATH=/work/src python -m ecotyper.cli --help'
```

## Check the installation

```bash
python -c "import ecotyper.workflows; print('ok')"
ecotyper-py --help
pytest -m "not e2e"          # unit parity tests against R fixtures (~1 min)
```

(example-data)=
## Example data and prebuilt models

The tutorials use the example data and the prebuilt **Carcinoma** and **Lymphoma** models that
ship with the R repository. Clone it next to this package:

```bash
git clone https://github.com/digitalcytometry/ecotyper.git
cd ecotyper
tar -xzf example_data/scRNA_CRC_data.txt.tar.gz -C example_data   # needed by tutorials 2 and 5
cd ..
```

The rest of the documentation assumes:

| Path | Contents |
|---|---|
| `ecotyper/example_data/` | input matrices and annotations of the tutorials |
| `ecotyper/EcoTyper/` | prebuilt models: `EcoTyper/Carcinoma/Carcinoma_Fractions/`, `EcoTyper/Lymphoma/Lymphoma_Fractions/` |

## Environment variables

| Variable | Effect |
|---|---|
| `ECOTYPER_COLLATION=C` | sort strings by bytes (like R with `LC_COLLATE=C`) instead of the ICU/UTF-8 collation |
| `ECOTYPER_LD=0` / `1` | force the long-double kernels off/on (default: on where `long double` is wider than `double`, i.e. x86-64) |
| `ECOTYPER_NO_FMA=1` | use `a*b + c` instead of fused multiply-add in the kernels where R's arm64 binaries use FMA; set it on x86-64 to match R builds without FMA (e.g. conda-forge R) |
| `ECOTYPER_CACHE` | where the long-double kernel library is compiled to (default: a temp folder) |
