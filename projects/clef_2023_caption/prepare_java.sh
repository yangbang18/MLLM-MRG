root=$(pwd)

java_root=data
fn="jre1.8.0_391"

mkdir -p $java_root
cd $java_root

if [[ ! -d $fn ]];
then
echo "install $fn..."
wget "https://s3.openi.org.cn/opendata/attachment/f/9/f99d8433-a502-4557-bac6-84e0887c8d51?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20231229%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20231229T111029Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22jre1.8.0_391.zip%22&X-Amz-Signature=59d770fbb488a1186d62eade3d4a0133920c462df893aa2c02a0c9e6860bf5e0" -O $fn.zip
unzip -q $fn.zip
rm $fn.zip
fi

cd $fn

JRE_HOME=$(pwd)
export PATH=${JRE_HOME}/bin:$PATH

# echo "export JRE_HOME=${JRE_HOME}" >> ~/.bashrc
# echo "export PATH=${JRE_HOME}/bin:$PATH" >> ~/.bashrc
# source ~/.bashrc

java -version

cd $root
echo "done"
