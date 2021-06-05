

Build Docker image:
```bash
$ docker build -t sapr_trainer_api -f app/main/configuration/docker/Dockerfile .
```

Run Docker container:
```bash
$ docker run -d --runtime=nvidia -v /home/manoel/Documents/workspace_talentify/sentiment_analisys_pull_requests/volume/model:/trainer/model -v /home/manoel/Documents/workspace_talentify/sentiment_analisys_pull_requests/volume/results:/trainer/results --name sapr_trainer_api --rm sapr_trainer_api
```