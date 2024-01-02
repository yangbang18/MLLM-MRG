root=$(pwd)

ckpt_root=data/checkpoints
mkdir -p $ckpt_root
cd $ckpt_root

if [[ ! -f eva_vit_g.pth ]];
then
echo "download $ckpt_root/eva_vit_g.pth..."
wget "https://s3.openi.org.cn/opendata/attachment/7/6/764f5dda-70db-4648-9782-68631c6e59cd?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20231229%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231229T111029Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22eva_vit_g.pth%22&X-Amz-Signature=eff624c60ba74499694b2ef4f71a7ae66930b87a8c293de553cfe545f9c8a52a" -O eva_vit_g.pth
fi

if [[ ! -f finetuned_eva_vit_g.pth ]];
then
echo "download $ckpt_root/finetuned_eva_vit_g.pth..."
wget "https://s3.openi.org.cn/opendata/attachment/f/d/fd69ca17-e563-4fd1-8bb0-e36e20d11a67?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20231229%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231229T111029Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22finetuned_eva_vit_g.pth%22&X-Amz-Signature=a783eacafb7f11ec3a9f0fc08e0c5f811286e2ab89da9ed2d26df699dc201934" -O finetuned_eva_vit_g.pth
fi

if [[ ! -d bert-base-uncased ]];
then
echo "download $ckpt_root/bert-base-uncased..."
wget "https://s3.openi.org.cn/opendata/attachment/4/d/4d96abb0-48f5-4bc6-bc87-904821ca493c?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20231229%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231229T111029Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22bert-base-uncased.zip%22&X-Amz-Signature=d13925972ee4bd7a9cfaa4d507f25bd77247488f4e33a45d4a207a2bba1d90e3" -O bert-base-uncased.zip
unzip bert-base-uncased.zip
rm bert-base-uncased.zip
fi

if [[ ! -d facebook_opt-2.7b ]];
then
echo "download $ckpt_root/facebook_opt-2.7b..."
wget "https://s3.openi.org.cn/opendata/attachment/6/3/631eeb6f-9f92-4feb-8fb0-10cea1aad50d?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20231229%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231229T111029Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22facebook_opt-2.7b.zip%22&X-Amz-Signature=42e04b9faf6cf0c787ad88627b642261647117af39e5efe0fe579ba111e74c77" -O facebook_opt-2.7b.zip
unzip facebook_opt-2.7b.zip
rm facebook_opt-2.7b.zip
fi

if [[ ! -d THUDM_chatglm-6b ]];
then
echo "download $ckpt_root/THUDM_chatglm-6b..."
wget "https://s3.openi.org.cn/opendata/attachment/a/3/a3533964-684a-41ab-91b9-c23b998bb4fb?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20240102%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240102T141438Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22THUDM_chatglm-6b.zip%22&X-Amz-Signature=67043b05823dd61067f0810a20d93830f6f48ef214effc1401d6249b0a7dfa69" -O THUDM_chatglm-6b.zip
unzip THUDM_chatglm-6b.zip
rm THUDM_chatglm-6b.zip
fi

cd $root
echo "done"
