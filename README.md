# Interactions-QUBO-optim



# everything in one go (default: g05_60)
python -m benchmarks.run

# pick datasets, override hyperparameters
python -m benchmarks.run g05_80 --steps 6000 --xi0 0.04 --repeats 3

# or step by step
python -m benchmarks.download g05_100
python -m benchmarks.benchmark g05_100 --device cuda
python -m benchmarks.visualize --latest