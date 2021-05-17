HASH=$(date | sha256sum | cut -c1-4)
read -p 'which server: ' server
if [ ${server} -eq 53 ];then
docker run -it \
--rm \
--runtime=nvidia \
--name kaggle-str-${HASH} \
--shm-size=128g \
-u $(id -u):$(id -g) \
-v /home/minki/kaggle/rsna-str-embolism:/workspace \
-v /nfs3/minki/kaggle/rsna-str-embolism:/nfs3/minki/kaggle/rsna-str-embolism \
-v /data/minki/kaggle/RSNA-STR_Pulmonary_Embolism_Detection:/nfs3/medical-image/RSNA-STR_Pulmonary_Embolism_Detection \
-w /workspace \
minki/cxr:v1.0
elif [ ${server} -eq 51 ];then
docker run -it \
--rm \
--runtime=nvidia \
--name kaggle-str-${HASH} \
--shm-size=128g \
-u $(id -u):$(id -g) \
-v /home/minki/kaggle/rsna-str-embolism:/workspace \
-v /nfs3/minki/kaggle/rsna-str-embolism:/nfs3/minki/kaggle/rsna-str-embolism \
-v /data2/beomheep/kaggle_emboli:/nfs3/medical-image/RSNA-STR_Pulmonary_Embolism_Detection \
-w /workspace \
minki/cxr:v1.0
fi


# "/nfs3/medical-image/RSNA-STR_Pulmonary_Embolism_Detection"cd 