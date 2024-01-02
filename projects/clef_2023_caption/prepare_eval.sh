root=$(pwd)

ckpt_root=data/checkpoints
mkdir -p $ckpt_root
cd $ckpt_root

if [[ ! -d microsoft_deberta-xlarge-mnli ]];
then
echo "download $ckpt_root/microsoft_deberta-xlarge-mnli for calculating BERTScore..."
wget "https://s3.openi.org.cn/opendata/attachment/a/3/a39380cf-f06b-49e7-8dfc-23b4a77ff191?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20231230%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231230T084327Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22microsoft_deberta-xlarge-mnli.zip%22&X-Amz-Signature=6669d249b1972df068c62126de9a1bd18fe8a65d28e064f7a942e758dfd4c4e3" -O microsoft_deberta-xlarge-mnli.zip
unzip microsoft_deberta-xlarge-mnli.zip
rm microsoft_deberta-xlarge-mnli.zip
fi

if [[ ! -d lucadiliello_BLEURT-20 ]];
then
echo "download $ckpt_root/lucadiliello_BLEURT-20 for calculating BLEURT..."
wget "https://s3.openi.org.cn/opendata/attachment/f/5/f5224f75-d8ad-4486-9de6-6b728acadb22?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20231230%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231230T103736Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22lucadiliello_BLEURT-20.zip%22&X-Amz-Signature=513baa13180f68c0e6a4faf947d13061bef57c03d4c500a79752c715e001fb35" -O lucadiliello_BLEURT-20.zip
unzip lucadiliello_BLEURT-20.zip
rm lucadiliello_BLEURT-20.zip
fi

pip install bert_score==0.3.13
pip install git+https://github.com/lucadiliello/bleurt-pytorch.git
pip install git+https://github.com/openai/CLIP.git
pip install scikit-learn

cd $root
echo "done"
