# Nd=(200000 400000 800000 1600000 2500000 4000000 20000000)
# Dim=(1000 500 250 125 80 50 10)
# Split=(1000 1000 10000 10000 10000 10000 10000)
# First=(1 0 0 0 0 0 0)

Nd=(200000 200000 200000 200000 200000 200000 200000)
subNd=(200000 150000 100000 50000 20000 10000 5000 1000)
Dim=(1000 500 250 125 80 50 10 3)
Split=(1000 1000 10000 10000 10000 10000 10000)
First=(1 0 0 0 0 0 0 0)
rm -rf out/performance/
mkdir -p out/performance/synthetic/ out/performance/mnist/

echo "Starting Performance test..."

for i in $(seq -w 0 7); do
    rm -rf data/performance/
    mkdir -p data/performance/
    echo "Dataset with Nd: ${Nd[$i]} and Dim: ${Dim[$i]}"
    python scr/data_scr/synthetic_generation.py -split ${Split[$i]} -sz ${Nd[$i]} -dim ${Dim[$i]} -dir performance
    echo "Dataset completed"
    
    echo "Starting test for current dataset..."
    for j in $(seq -w 0 7);do
        echo "Running for Nd: ${subNd[$j]}"
        bin/performance -d 1 -sz ${subNd[$j]} -dim ${Dim[$i]} -samp 5000 -burn 1000 -lag 500 -tune 0 -first ${First[$i]}
    done
    echo "Test for current dataset completed..."
done