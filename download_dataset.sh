echo "Downloading dataset"
#wget http://neghvar.cs.umass.edu/public_data/parsenet/data.zip
wget http://neghvar.cs.umass.edu/public_data/parsenet/predictions.h5
echo "unzipping"
#unzip data.zip
mkdir logs
mkdir logs/results
mkdir logs/results/parsenet_with_normals.pth
mkdir logs/results/parsenet_with_normals.pth/results
mv predictions.h5 logs/results/parsenet_with_normals.pth/results/predictions.h5
