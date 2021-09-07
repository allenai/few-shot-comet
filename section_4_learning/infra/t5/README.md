# minimal_t5

## vm setup

    sudo apt-get -y install git vim build-essential tmux wget 
    wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh -O ~/anaconda.sh
    bash ~/anaconda.sh -b -p $HOME/anaconda3
    echo "export PATH=/home/\$USER/anaconda3/bin:\$PATH" >>.profile
    echo "alias ll='ls -aGhl --color=auto'" >> .profile
    source ~/.profile

(optionally, get your .rc, .conf etc.)
    
    wget https://gist.githubusercontent.com/keisks/ad875ccc618cb8ad36ccc693e778ff23/raw/e875dc073df0ebd244cfe8b3adf8e4aaad0e7b3f/.tmux.conf
    wget https://raw.githubusercontent.com/git/git/master/contrib/completion/git-completion.bash
    curl -L https://raw.github.com/git/git/master/contrib/completion/git-prompt.sh > ~/.bash_git
    echo "source ~/.bash_git" >> .profile
    echo "source ~/git-completion.bash" >> .profile
    echo "GIT_PS1_SHOWDIRTYSTATE=true" >> .profile
    echo "export PS1='\[\033[35m\]\u\[\033[31m\]@\h\[\033[00m\]:\[\033[33m\]\W\[\033[32m\]\$(__git_ps1)\[\033[00m\]\\$ ' " >> .profile
    source ~/.profile

(optionally, add [Cloud Monitoring agent](https://cloud.google.com/monitoring/agent/installation#agent-install-debian-ubuntu))

    curl -sSO https://dl.google.com/cloudagents/add-monitoring-agent-repo.sh
    sudo bash add-monitoring-agent-repo.sh
    sudo apt-get update
    sudo apt-cache madison stackdriver-agent
    sudo apt-get install -y 'stackdriver-agent=6.*'
    sudo service stackdriver-agent start

## Set up ctpu (and exp env variables)

    tmux
    export ZONE="europe-west4-a"
    wget https://dl.google.com/cloud_tpu/ctpu/latest/linux/ctpu && chmod a+x ctpu
    sudo cp ctpu /usr/local/bin
    gcloud auth login
    (Note: this requires manual auth login, enter verification code)
    gcloud auth application-default login
    (Note: this requires manual auth login, enter verification code)

    gcloud config set compute/zone $ZONE
    gcloud compute tpus list --zone=$ZONE

## Set up conda

    export CONDA_ENV_NAME=YOUR_CONDA_ENV_NAME [change here]
    conda update -n base -c defaults conda
    conda create -n $CONDA_ENV_NAME python=3.7
    conda init bash
    source ~/.bashrc
    conda activate $CONDA_ENV_NAME
    pip install -r requirements.txt

## Get data ready
Put train/dev/test.tsv into the following (for example)

    gs://ai2-tpu-europe-west4/projects/keisukes/example/t5-data/task1
    (for multiple datasets mixture, create task2, task3, etc.)

## training

    python t5_run.py train --func f --model_dir_to_save gs://ai2-tpu-europe-west4/projects/keisukes/example/t5-model/testrun-small --num_steps 100 --model_size small --mixture_name new --tpu_name temp-tpu-1 --mixture_dir gs://ai2-tpu-europe-west4/projects/keisukes/example/t5-data/task1 --prev_trained_model gs://t5-data/pretrained_models/small

## prediction

    python t5_run.py predict --prev_trained_model gs://ai2-tpu-europe-west4/projects/keisukes/example/t5-model/testrun-small --model_size small --input_csv gs://ai2-tpu-europe-west4/projects/keisukes/example/t5-data/task1/test.tsv --output_csv ai2-tpu-europe-west4/projects/keisukes/example/t5-data/task1/pred_test.tsv --tpu_name temp-tpu-1

