# Added by Yang B.

gpu=$1
config=$2
options=$3

config=${config:-projects/clef_2023_caption/configs/364_ViTg_FT1_ChatGLM_ptuning4.yaml}
options=${options:-}

if [[ -d data/jre1.8.0_391 ]];
then
    export JRE_HOME=data/jre1.8.0_391/jre
    export PATH=${JAVA_HOME}/bin:$PATH
fi

if [[ $gpu = 1 ]];
then
    python train.py --cfg-path ${config} $options
else
    python -m torch.distributed.run --nproc_per_node=${gpu} train.py --cfg-path ${config} $options
fi
