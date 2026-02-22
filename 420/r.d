
# docker build -t tfq-image-classifier .
# # Mount your data folder
# docker run -it --rm -v $(pwd)/data:/app/data tfq-image-classifier \
#   --class0_dir /Users/enrique/Desktop/fun_repos/quantum-mnist/data/zooplankton_0p5x/dinobryon/training_data \
#   --class1_dir /Users/enrique/Desktop/fun_repos/quantum-mnist/data/zooplankton_0p5x/asplanchna/training_data \
#   --epochs 50