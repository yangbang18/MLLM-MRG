root=$(pwd)

dataset_root=data/clef_2023_caption
mkdir -p $dataset_root
cd $dataset_root

if [[ ! -f train.json ]];
then
echo "download $dataset_root/train.json..."
wget "https://s3.openi.org.cn/opendata/attachment/d/7/d7f3534a-91e4-42c9-8b46-abacf67deb30?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20231229%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231229T085254Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22train.json%22&X-Amz-Signature=7ae889e9538a5db256bf82c1569c68cb259ab7292588884330fe5f1b31eb4f89" -O train.json
fi

if [[ ! -f val.json ]];
then
echo "download $dataset_root/val.json..."
wget "https://s3.openi.org.cn/opendata/attachment/d/9/d9f24499-d0d7-44dd-bfac-7912f64391da?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20231229%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231229T085254Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22val.json%22&X-Amz-Signature=97a74f23f40143f662ff751f156ee43987d4a7a82e583443b9c8f6b82676c5f7" -O val.json
fi

if [[ ! -f val_gt.json ]];
then
echo "download $dataset_root/val_gt.json..."
wget "https://s3.openi.org.cn/opendata/attachment/6/8/68de389d-4fe9-4461-80e3-2de86304e24c?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20231229%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231229T090833Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22val_gt.json%22&X-Amz-Signature=e07f76f65f50f6ebee74542606b7b5c3f2bae3bc7617247ed7e2026a5067b907" -O val_gt.json
fi

if [[ ! -f test.json ]];
then
echo "download $dataset_root/test.json..."
wget "https://s3.openi.org.cn/opendata/attachment/f/0/f0465de9-1c79-4f61-9662-cad5005e55a8?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20231229%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231229T085254Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22test.json%22&X-Amz-Signature=8fb42c9cab2fb99acafe44b980a9df445972f28b74728a751f981ba05f3b7072" -O test.json
fi

if [[ ! -d images ]];
then
echo "download $dataset_root/images.zip..."
wget "https://s3.openi.org.cn/opendata/attachment/5/8/586ae746-140f-418a-b2fa-ae224ea7845f?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20231229%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231229T090833Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22images.zip%22&X-Amz-Signature=fe21fb5fae9eb8065581ffb16d0a35c61414bfe68116dbe54f057241f713aa00" -O images.zip

echo "unzip $dataset_root/images.zip..."
unzip -q images.zip
rm images.zip
fi

cd $root
echo "done"
