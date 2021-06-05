

Build Docker image:
```bash
$ docker build -t sapr_trainer_api -f app/main/configuration/docker/Dockerfile .
```

# Enviroments parameters 
Parâmetros previstos a serem informados ao rodar o container:

* NUMBER_OF_CLASSES (obrigatório)
* EARLY_STOPPING
* EPOCHS
* TRAIN_PATH_FILE (obrigatório)
* VAL_PATH_FILE (obrigatório)
* TEST_PATH_FILE (obrigatório)
* MAX_LEN
* BATCH_SIZE
* FEATURE_NAME (obrigatório)
* TARGET_NAME (obrigatório)
* NUM_WORKERS
* BASELINE_PATH
* CHECKPOINT_PATH (obrigatório)
* LEARNING_RATE

Para o checkpoint do fine-tunning do modelo e arquivo com os resultados no dataset de testes deverão ser salvos
em duas pastas configuradas como volume para ter acesso de fora do container.

* /trainer/model
* /trainer/results

Para rodar o container é esperado o uso de GPU, caso não tenha configurado o docker para rodar com o container da NVIDIA,
o tutorial no link (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) deve ser usado como refeência.

Run Docker container:
```bash
$ docker run -d --runtime=nvidia -v /home/manoel/Documents/workspace_talentify/sentiment_analisys_pull_requests/volume/model:/trainer/model\
 -v /home/manoel/Documents/workspace_talentify/sentiment_analisys_pull_requests/volume/results:/trainer/results\
 -e BASELINE_PATH="./baseline" -e CHECKPOINT_PATH="./model" -e TRAIN_PATH_FILE="./data/train.csv"\
 -e VAL_PATH_FILE="./data/val.csv" -e TEST_PATH_FILE="./data/test.csv" -e NUMBER_OF_CLASSES=3\
 -e FEATURE_NAME="text" -e TARGET_NAME="airline_sentiment" -e EARLY_STOPPING=3 --name sapr_trainer_api\--rm sapr_trainer_api
```