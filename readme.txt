-----------7/13/24 testing igor's kid routine --------
requires tensorflow 2.x with keras.
conda activate openmmlab

the original mmdgan use tf1.3 so it is easy to run in docker container

debugging:
File "kid.py", line 88, in <module>
    inl = np.array(images1)[:32, :, :, :]
IndexError: too many indices for array: array is 1-dimensional, but 4 were indexed

-------------------------7/7/24 core.mmd no attribute __kernel----------------------------------------
  File "/workspace/gan/core/model.py", line 160, in set_loss
    kernel = getattr(mmd, '_%s_kernel' % self.config.kernel)
AttributeError: module 'core.mmd' has no attribute '__kernel'

-------------------------7/6/24 dataset issue----------------------------------------
cd gan; python3.6 main.py --data_dir ../data

log see:
	samples_mmd/mmd_test/cifar1064x64_dcgan_dc_d5-5-1_32_32_lr0.00010000_bn/log.txt
	No such file or directory: './data/cifar10/data_batch_1'
Log file: <_io.TextIOWrapper name='samples_mmd/mmd_test/mnist64x64_dcgan_dc_d5-5-1_32_28_lr0.00010000_bn/log.txt' mode='w' encoding='ANSI_X3.4-1968'>

fixed:
	~/Documents/igor/dcgan/dcgan$ cp MNIST/raw/* ~/Documents/igor/MMD-GAN/gan/data/mnist
-------------------------7/5/24 docker setup tf1.4 py3.6----------------------------------------
docker pull jwang3vsu/tf14py36:latest
cd ~/Documents/igor/MMD-GAN
docker compose -f docker-compose-tf1_3.yaml up top

docker exec -it mmd-gan-top bash
	cd /workspace
	pip3.6 install -r docker/requirements.txt
	download mnist dataset to data/
		mkdir data
		https://drive.google.com/file/d/1V35qidHrK9bThiNAdyYdQdJS3gRUEFxl/view?usp=sharing
	./mnist.sh

testing cifar10.sh
	download cifar10 data using dcgan main.py and copy to .data/cifar10
		dcgan$ python main.py --dataset cifar10 --dataroot /data/cifar10
	inside docker:
		./cifar10.sh
		
docker image build:
	~/Documents/phoenix_note2/docker/tf14$ vi readme_tf13docker.txt 

-----------------------------------------------------------------------------
  pip3.6 install lmdb
  pip3.6 install Pillow
  pip3.6 install matplotlib
  pip3.6 install sklearn
  pip3.6 install tqdm
  python3.6 main.py 

