# <span style="color:blue">PROJETO: Análise de Sentimentos - Pull Requests - API</span>

Este é o projeto da API do modelo de análise de sentimentos que foi publicado utilizando o framework [BentoML](https://docs.bentoml.org/en/latest/index.html).

Para rodar o projeto o comando abaixo precisa ser executado:

### Instalação do BentoML e pacotes de dependências do modelo
```bash
$ pip install -r app/main/configuration/requirements.txt
```


### Publicação do modelo no BentoML
```bash
$ python main.py ../volume/model/
```
O valor <i>../volume/model/</i> é o parâmetro do caminho onde o modelo treinado foi salvo.
No final dessa execução o programa vai imprimir o caminho de onde o BentoML salvou a API do modelo, exemplo: <i>/home/manoel/bentoml/repository/SentimentService/20210607094602_CB7D26</i>.

Além disso imprime também o ID do serviço publicado, exemplo: <i>BentoService bundle 'SentimentService:20210607094602_CB7D26'</i>.

Esse valor é o que deve ser passado para os próximos comandos quando executados.

### Servindo o modelo via REST API
```bash
$ bentoml serve SentimentService:20210607094602_CB7D26
```

### Criando um container Docker do modelo
```bash
$ bentoml containerize SentimentService:20210607094602_CB7D26 -t sapr_service_api
```

### Subindo o container Docker do modelo
```bash
$ docker run -d -p 5000:5000 sapr_service_api:20210607094602_CB7D26
```

### Subindo o container Docker do modelo - Imagem DockerHub
```bash
$ docker run -d -p 5000:5000 verissimomanoel/sapr_service_api
```