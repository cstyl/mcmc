Nd=(400000 800000 1600000 3200000 5000000 8000000 80000000)
Dim=(1000 500 250 125 80 50 10)
Split=(1000 1000 10000 10000 10000 10000 10000)
# Nd=(1000 2000)
# Dim=(5 3)
# Split=(10 10)
First=(1 0 0 0 0 0 0)

rm -rf out/performance/
mkdir -p out/performance/synthetic/ out/performance/mnist/

echo "Starting Performance test..."

for i in $(seq -w 0 6); do
    rm -rf data/performance/
    mkdir -p data/performance/
    echo "Dataset with Nd: ${Nd[$i]} and Dim: ${Dim[$i]}"
    python scr/data_scr/synthetic_generation.py -split ${Split[$i]} -sz ${Nd[$i]} -dim ${Dim[$i]} -dir performance
    echo "Dataset completed"
    
    echo "Starting test for current dataset..."
    bin/performance -d 1 -sz ${Nd[$i]} -dim ${Dim[$i]} -samp 5000 -burn 1000 -lag 500 -tune 0 -first ${First[$i]}
    echo "Test for current dataset completed..."
done