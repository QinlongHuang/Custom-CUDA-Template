# How to run these scripts?
```bash
# Install python package
pip install -e .

# Manually Build
cmake -S . -Bbuildcm -Gninja
cmake --build buildcm

# Run!
python test.py
```